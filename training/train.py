import os
import jax
jax.distributed.initialize()

import tensorflow as tf
from functools import partial
import jax.numpy as jnp
from jax.experimental import multihost_utils, mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental.multihost_utils import process_allgather
import optax
import numpy as np
import tqdm
import traceback
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
import random
import copy
import datetime
from einops import rearrange
import flax
from flax.core import freeze, unfreeze, FrozenDict
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from etils import epath

from bageljax.common.wandb import WandBLogger
from bageljax.common.common import TrainState, ModuleDict, nonpytree_field
from bageljax.common.optimizers import make_optimizer
from bageljax.common.typing import Batch, PRNGKey
from bageljax.data.dataset import Dataset
from bageljax.utils.timer_utils import Timer
from bageljax.utils.jax_utils import host_broadcast_str, create_sharding, add_batch_sharding_constraint, enforce_sharding_constraints, initialize_compilation_cache
from bageljax.model.vocabulary import TokenEmbedder, LogitsHead
from bageljax.model.vision_encoder import VisionEncoder
from bageljax.model.mixture_of_transformers import MixtureOfTransformers
from bageljax.model.tokenizer import Qwen2Tokenizer, add_special_tokens

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_list('tag', list(), 'Name of experiment')
flags.DEFINE_string('group', None, 'Group of the wandb experiments')
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def print_green(message):
    print(f"\033[92m{message}\033[0m")

def main(_):
    # Initialize JAX compilation cache on a shared GCS path so all workers
    # load compiled XLA programs on subsequent runs instead of recompiling.
    initialize_compilation_cache("gs://raymond-us-west1/jax_compilation_cache")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": f"value_function",
            "exp_descriptor": FLAGS.exp_name,
            "tag": FLAGS.tag,
            "group": FLAGS.group,
        }
    )
    unique_id = "{time}".format(
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    unique_id = host_broadcast_str(unique_id)
    wandb_config.update({"unique_identifier": unique_id})
    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_config.project,
        f"{FLAGS.exp_name}_{unique_id}",
    )
    if jax.process_index() == 0:
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=FLAGS.config.to_dict(),
            debug=FLAGS.debug,
        )

    # load datasets
    print("Initializing data loader...")

    # Use different dataset seeds for each worker
    dataset_seed = FLAGS.config.seed + jax.process_index()

    # Create the training dataset
    train_data = Dataset(
        data_paths=FLAGS.config.dataset_kwargs["data_paths"],
        seed=dataset_seed,
        batch_size=FLAGS.config.dataset_kwargs["batch_size"],
        shuffle_buffer_size=FLAGS.config.dataset_kwargs["shuffle_buffer_size"],
        num_parallel_calls=FLAGS.config.dataset_kwargs["num_parallel_calls"],
        train=True,
    )
    train_data_iter = train_data.iterator()
    print("Data loader initialized")

    # Create the language tokenizer
    tokenizer_load_path = FLAGS.config.tokenizer_load_path
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Function to convert text instruction strings into padded tokens, and adds to the batch 
    # these tokens as well as the token masks and text rope IDs
    def tokenize_and_pad(batch):
        # What this function is going to do is rewrite the language instruction into:
        # "How many timesteps away is the robot from successfully completing the following language instruction:\n\n<language_instruction>\n\nDistance:"
        # and then tokenize. It will also left-pad the text to the max length, and assign RoPE IDs.

        PAD_TOKEN_ID = 0 # it doesn't really matter what this is
        MAX_PROMPT_LENGTH = FLAGS.config.dataset_kwargs["max_prompt_length"]

        B = batch["image"].shape[0]
        batch_tokenized_language = []
        batch_masks = []
        batch_text_rope_ids = []
        for i in range(B):
            prompt = "How many timesteps away is the robot from successfully completing the following language instruction:\n\n"
            prompt += batch["language_instruction"][i].decode("utf-8").strip()
            prompt += "\n\nDistance:"
            prompt_text_ids = tokenizer.encode(prompt)
            
            # Add bos and eos tokens
            prompt_text_ids = [new_token_ids['bos_token_id']] + prompt_text_ids + [new_token_ids['eos_token_id']]

            # Also add the bos token at the end to start value prediction
            prompt_text_ids = prompt_text_ids + [new_token_ids['bos_token_id']]

            non_pad_tokens_in_prompt = len(prompt_text_ids)
            assert non_pad_tokens_in_prompt < MAX_PROMPT_LENGTH

            # Pad on the left
            num_pad_tokens = MAX_PROMPT_LENGTH - non_pad_tokens_in_prompt
            prompt_text_ids = [PAD_TOKEN_ID] * num_pad_tokens + prompt_text_ids
            prompt_text_ids = np.array(prompt_text_ids, dtype=np.int32)
            masks = np.concatenate([np.zeros((num_pad_tokens,), dtype=bool), np.ones((non_pad_tokens_in_prompt,), dtype=bool)]) # false means padding, true means valid text token
            text_rope_ids = np.concatenate([
                np.zeros((num_pad_tokens,), dtype=np.int32), # the rope IDs for pad tokens doesn't matter
                np.arange(non_pad_tokens_in_prompt, dtype=np.int32) + 1, # start from 1, since the image comes before
            ])

            batch_tokenized_language.append(prompt_text_ids)
            batch_masks.append(masks)
            batch_text_rope_ids.append(text_rope_ids)

        batch_tokenized_language = np.stack(batch_tokenized_language)
        batch_masks = np.stack(batch_masks)
        batch_text_rope_ids = np.stack(batch_text_rope_ids)

        del batch["language_instruction"]
        batch["text_tokens"] = batch_tokenized_language
        batch["text_token_masks"] = batch_masks
        batch["text_rope_ids"] = batch_text_rope_ids

        return batch

    # Initialize rng from config seed
    rng = jax.random.PRNGKey(FLAGS.config.seed)

    # We're starting model creation; set enforce sharding constraints to false for this phase
    enforce_sharding_constraints(False)

    # Initialize the main model
    print("Initializing Bagel Value Function model...")

    networks = {
        "token_embedder": TokenEmbedder(),
        "vision_encoder": VisionEncoder(),
        "mixture_of_transformers": MixtureOfTransformers(),
        "logits_head": LogitsHead(),
    }

    model_def = ModuleDict(networks)

    # create optimizer
    tx, lr_schedule = make_optimizer(
        learning_rate=FLAGS.config.policy_kwargs["learning_rate"],
        weight_decay=FLAGS.config.policy_kwargs["weight_decay"],
        beta2=FLAGS.config.policy_kwargs["b2"],
        eps=FLAGS.config.policy_kwargs["eps"],
        cosine_decay_steps=FLAGS.config.policy_kwargs["decay_steps"],
        warmup_steps=FLAGS.config.policy_kwargs["warmup_steps"],
        clip_grad_norm=1.0,
        return_lr_schedule=True,
    )

    def init_fn(rng):
        rng, init_rng = jax.random.split(rng)

        # For init, let's pick reasonable values of some of the input parameters
        B = 1 # batch size
        H, W = 672, 672 # image height and width, 672 is divisible by 14 and 16 ---> no longer needs to be divisible by 16
        llm_hidden_dim = 3584 # LLM hidden dimension
        L = 256 # needs to be a multiple of 128 for flash attention

        params = model_def.init({'params': init_rng},
                                    token_embedder = [
                                        jnp.zeros((B, L), dtype=jnp.int32),
                                    ],
                                    vision_encoder = [
                                        jnp.zeros((B, H, W, 3), dtype=jnp.bfloat16),
                                    ],
                                    mixture_of_transformers = [
                                        jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                        jnp.zeros((B, L), dtype=jnp.int32),
                                        jnp.zeros((B, 1, L, L), dtype=jnp.bfloat16),
                                    ],
                                    logits_head = [
                                        jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                    ],
                                )["params"]
        rng, create_rng = jax.random.split(rng)
        train_state = TrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )
    
        return train_state

    rng, key = jax.random.split(rng)
    train_state_shape = jax.eval_shape(init_fn, key)

    # Create sharding and train_state
    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding("fsdp", train_state_shape)
    rng, key = jax.random.split(rng)
    train_state = jax.jit(init_fn, out_shardings=train_state_sharding)(key)

    # Create checkpointer object (used for all subsequent checkpoint loading)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())

    # Load from pre-trained Bagel checkpoint
    restored_params = checkpointer.restore(FLAGS.config.pretrained_bagel_path, train_state.params)
    train_state = train_state.replace(params=restored_params)

    # Update target params
    train_state = train_state.target_update(tau=1.0)

    print("Loaded from pre-trained Bagel")

    # Load from a previous checkpoint if necessary
    if FLAGS.config.get("resume_path", None) is not None:
        print("Loading from previous checkpoint...")
        train_state = checkpointer.restore(
            FLAGS.config.resume_path,
            train_state,
        )
        print("Loaded from previous checkpoint")

    # Print total number of parameters of the model
    print("Total model parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(train_state.params)]))

    # Create auxiliary objects for use by update function
    config = flax.core.FrozenDict(
        dict(
            num_buckets=FLAGS.config.policy_kwargs["num_buckets"],
            discount_factor=FLAGS.config.policy_kwargs["discount_factor"],
            target_update_rate=FLAGS.config.policy_kwargs["target_update_rate"],
        )
    )

    # Define update function
    @partial(
        jax.jit,
        in_shardings=(train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, no_shard),
        static_argnames=["config", "lr_schedule"],
    )
    def update(train_state, batch, config, lr_schedule):
        def loss_fn(params, rng):
            # Prepare the vit tokens
            image = jnp.astype(batch["image"], jnp.float32) / 127.5 - 1
            image = jnp.astype(image, jnp.bfloat16)
            image = add_batch_sharding_constraint(image, where="image before vit")
            
            # Pass through ViT
            pre_llm_vit_tokens = train_state.apply_fn(
                {"params": params},
                img=image,
                name="vision_encoder",
            )
            pre_llm_vit_tokens = add_batch_sharding_constraint(pre_llm_vit_tokens, where="image after vit")

            # Let's embed the start and end image tokens
            image_special_token_ids = jnp.array([new_token_ids['start_of_image'], new_token_ids['end_of_image']], dtype=jnp.int32)
            image_special_token_embeds = train_state.apply_fn(
                {"params": params},
                token_ids=image_special_token_ids[None, :],
                name="token_embedder",
            )
            image_special_token_embeds = jnp.tile(image_special_token_embeds, (pre_llm_vit_tokens.shape[0], 1, 1))
            image_special_token_embeds = add_batch_sharding_constraint(image_special_token_embeds, where="image_special_token_embeds")

            # concat the special image tokens
            pre_llm_vit_tokens = jnp.concatenate([image_special_token_embeds[:, 0:1], pre_llm_vit_tokens, image_special_token_embeds[:, 1:2]], axis=1)
            pre_llm_vit_tokens = add_batch_sharding_constraint(pre_llm_vit_tokens, where="pre_llm_vit_tokens")

            # embed the text tokens
            text_embeds = train_state.apply_fn(
                {"params": params},
                token_ids=batch["text_tokens"],
                name="token_embedder",
            )
            text_embeds = add_batch_sharding_constraint(text_embeds, where="text_embeds")

            # Concat everything along sequence dimension
            full_seq = jnp.concatenate([pre_llm_vit_tokens, text_embeds], axis=1)
            full_seq = add_batch_sharding_constraint(full_seq, where="full_seq")

            # Prepare full seq rope IDs
            image_rope_ids = jnp.zeros((pre_llm_vit_tokens.shape[1],), dtype=jnp.int32)[None, :]
            image_rope_ids = jnp.tile(image_rope_ids, (full_seq.shape[0], 1))

            full_seq_rope_ids = jnp.concatenate([image_rope_ids, batch["text_rope_ids"]], axis=1)
            full_seq_rope_ids = add_batch_sharding_constraint(full_seq_rope_ids, where="full_seq_rope_ids")

            # Now prepare attention masks
            L = full_seq.shape[1]
            # Start with a causal mask
            causal = jnp.tril(jnp.ones((L, L), dtype=bool))
            # Enable full self-attention for vit tokens
            arr = jnp.concatenate([jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=bool), jnp.zeros((L - pre_llm_vit_tokens.shape[1]), dtype=bool)])
            self_attention_mask = jnp.matmul(arr[:, None], arr[None, :])
            block_causal = causal | self_attention_mask
            # Now tile to include batch dimension
            block_causal = jnp.tile(block_causal[None, ...], (full_seq.shape[0], 1, 1))
            block_causal = add_batch_sharding_constraint(block_causal, where="block_causal")
            # padding sees nothing, and is seen by nothing
            padding = jnp.concatenate([jnp.zeros((full_seq.shape[0], pre_llm_vit_tokens.shape[1]), dtype=bool), jnp.logical_not(batch["text_token_masks"])], axis=1)
            padding = add_batch_sharding_constraint(padding, where="padding")
            allowed_attention = jnp.where(padding[:, :, None] | padding[:, None, :], False, block_causal)
            attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
            attn_bias = attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol
            attn_bias = add_batch_sharding_constraint(attn_bias, where="attn_bias")

            # Now feed through the LLM
            post_llm_seq = train_state.apply_fn(
                {"params": params},
                x=full_seq,
                rope_pos_ids=full_seq_rope_ids,
                attn_bias=attn_bias,
                name="mixture_of_transformers",
            )
            post_llm_seq = add_batch_sharding_constraint(post_llm_seq, where="post_llm_seq")

            # Extract just the last token
            post_llm_value_token = post_llm_seq[:, -1:]

            # Feed into LLM head
            value_logits = train_state.apply_fn(
                {"params": params},
                hidden_states=post_llm_value_token,
                name="logits_head",
            )
            # this should have shape (B, 1, 512)
            assert value_logits.shape == (post_llm_seq.shape[0], 1, 512)
            assert value_logits.dtype == jnp.bfloat16
            value_logits = add_batch_sharding_constraint(value_logits, where="value_logits")
            # get rid of singleton dimension
            value_logits = value_logits[:, 0]

            # Construct cross-entropy targets
            discounted_distances = jnp.power(config["discount_factor"], batch["distance"]-1) # we subtract 1 bc we know that min(batch["distance"])==1
            # this will have shape (B,), dtype float32, and contain values from 0 to 1
            assert discounted_distances.shape == (post_llm_seq.shape[0],)
            assert discounted_distances.dtype == jnp.float32
            bucket_ids = jnp.clip((discounted_distances * config["num_buckets"]).astype(jnp.int32), min=0, max=config["num_buckets"]-1) # values in [0, config["num_buckets"]-1]
            value_targets = config["num_buckets"] - bucket_ids - 1 # reverse so that closer distances have lower bucket ids
            value_targets = add_batch_sharding_constraint(value_targets, where="value_targets")

            # Critical: loss should be computed in float32
            value_logits = jnp.astype(value_logits, jnp.float32)

            loss = optax.losses.softmax_cross_entropy_with_integer_labels(value_logits, value_targets)
            loss = add_batch_sharding_constraint(loss, where="loss before pmean")
            loss = jnp.mean(loss)

            # For logging, let's compute the accuracy in addition to the loss
            argmax_prediction_matches = jnp.argmax(value_logits, axis=-1) == value_targets
            accuracy = jnp.mean(argmax_prediction_matches)

            log_dict = {
                "loss": loss,
                "accuracy": accuracy,
            }

            return loss, log_dict

        # compute gradients and update params
        new_state, info = train_state.apply_loss_fns(
            loss_fn, has_aux=True,
        )

        # log learning rates
        info["lr"] = lr_schedule(train_state.step)

        # target update
        new_state = new_state.target_update(1 - config["target_update_rate"])

        return new_state, info
        
    # Starting train loop and thus compilation of the main graph; set enforce sharding constraints to true
    enforce_sharding_constraints(True)

    # Prepare for checkpointing the model
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    def save_ckpt(save_dir: str, step: int, train_state):
        """Asynchronous, multihost-safe save. Call from ALL hosts, outside any mesh context."""
        step_dir = epath.Path(save_dir) / f"{int(step):08d}"
        # Describe how to save each leaf (dtype/shape/metadata) without touching array data.
        save_args = orbax_utils.save_args_from_target(train_state)
        # Returns immediately; writes happen in the background.
        checkpointer.save(step_dir, args=ocp.args.StandardSave(train_state, save_args=save_args))

    # Synchronize all workers before entering the training loop.
    # This ensures no worker is ahead of others due to variable initialization time.
    multihost_utils.sync_global_devices("pre_train_loop")

    # Train loop
    timer = Timer()
    starting_train_step = int(jax.device_get(train_state.step))
    for i in tqdm.tqdm(range(starting_train_step, FLAGS.config.num_steps)):
        try:
            timer.tick("total")

            timer.tick("dataset")
            batch = next(train_data_iter)
            batch = tokenize_and_pad(batch)
            batch = shard_data(batch)
            timer.tock("dataset")

            timer.tick("train")
            train_state, update_info = update(train_state, batch, config, lr_schedule)
            timer.tock("train")

            timer.tock("total")

            step = i + 1

            if step % FLAGS.config.save_interval == 0 or step == 20:
                print_green(f"[process={jax.process_index()}] About to save checkpoint at step {step}")
                jax.block_until_ready(train_state)
                save_ckpt(save_dir, step, train_state)
                print_green(f"[process={jax.process_index()}] Checkpoint saved at step {step}")

            if step % FLAGS.config.log_interval == 0:
                jax.block_until_ready(train_state)
                if jax.process_index() == 0:
                    update_info = jax.device_get(update_info)
                    wandb_logger.log({"training": update_info}, step=step)
                    wandb_logger.log({"timer": timer.get_average_times()}, step=step)
                    
        except KeyboardInterrupt:
            break
        except tf.errors.OpError as e:
            # sometimes tfds will have trouble communicating with cloud storage bucket for some reason...
            print(f"Error in iteration {i}: {e}")
            print("Skipping to next iteration...")
            traceback.print_exc()

            # to deal with possible untocked timer counts
            timer.force_tock_everything()
            
            continue
    


if __name__ == "__main__":
    app.run(main)
