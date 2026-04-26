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
from bageljax.data.roboarena_dataset import Dataset as RoboArenaDataset
from bageljax.utils.timer_utils import Timer
from bageljax.utils.jax_utils import host_broadcast_str, create_sharding, add_batch_sharding_constraint, enforce_sharding_constraints, initialize_compilation_cache
from bageljax.model.vocabulary import TokenEmbedder, LogitsHead, ActionProjector
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

    dk = FLAGS.config.dataset_kwargs
    ra_dk = FLAGS.config.roboarena_dataset_kwargs
    total_batch_size = int(dk["batch_size"])
    use_roboarena = bool(FLAGS.config.get("use_roboarena", True))
    half_batch_size = total_batch_size // 2 if use_roboarena else total_batch_size

    # Reduce shuffle buffer size for each pipeline when mixing two sources to avoid OOM
    _min_shuffle = 2048
    if use_roboarena:
        shuffle_droid = max(_min_shuffle, int(dk["shuffle_buffer_size"]) // 2)
        shuffle_robo = max(_min_shuffle, int(ra_dk["shuffle_buffer_size"]) // 2)
        print(f"use_roboarena=True: shuffle_droid={shuffle_droid}, shuffle_robo={shuffle_robo}")
    else:
        shuffle_droid = max(_min_shuffle, int(dk["shuffle_buffer_size"]))
        print(f"use_roboarena=False: shuffle_droid={shuffle_droid} (DROID only)")

    # Create the training dataset
    train_data = Dataset(
        data_paths=dk["data_paths"],
        seed=dataset_seed,
        batch_size=half_batch_size,
        shuffle_buffer_size=shuffle_droid,
        num_parallel_calls=dk["num_parallel_calls"],
        train=True,
        action_chunk_size=dk["action_chunk_size"],
        action_joint_velocity_mean=dk.get("action_joint_velocity_mean"),
        action_joint_velocity_std=dk.get("action_joint_velocity_std"),
        action_norm_eps=float(dk.get("action_norm_eps", 1e-8)),
    )
    train_data_iter = train_data.iterator()
    print("DROID data loader initialized")

    roboarena_data_iter = None
    if use_roboarena:
        # Create the RoboArena dataset
        roboarena_train_data = RoboArenaDataset(
            data_paths=ra_dk["data_paths"],
            seed=dataset_seed,
            batch_size=half_batch_size,
            shuffle_buffer_size=shuffle_robo,
            num_parallel_calls=ra_dk["num_parallel_calls"],
            train=True,
            action_chunk_size=ra_dk["action_chunk_size"],
            action_joint_velocity_mean=ra_dk.get("action_joint_velocity_mean"),
            action_joint_velocity_std=ra_dk.get("action_joint_velocity_std"),
            action_norm_eps=float(ra_dk.get("action_norm_eps", 1e-8)),
        )
        roboarena_data_iter = roboarena_train_data.iterator()

    def merge_batches(droid_batch, robo_batch):
        """Build one training batch with half DROID + half RoboArena."""
        droid_bsz = droid_batch["image"].shape[0]
        robo_bsz = robo_batch["image"].shape[0]
        merged = {
            "image": np.concatenate([droid_batch["image"], robo_batch["image"]], axis=0),
            "language_instruction": np.concatenate(
                [droid_batch["language_instruction"], robo_batch["language_instruction"]],
                axis=0,
            ),
            "action/joint_velocity_chunk": np.concatenate(
                [
                    droid_batch["action/joint_velocity_chunk"],
                    robo_batch["action/joint_velocity_chunk"],
                ],
                axis=0,
            ),
            "distance": np.concatenate(
                [droid_batch["distance"], robo_batch["value_target"]], axis=0
            ),
            "is_roboarena": np.concatenate(
                [
                    np.zeros((droid_bsz,), dtype=bool),
                    np.ones((robo_bsz,), dtype=bool),
                ],
                axis=0,
            ),
        }
        return merged

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
        "action_projector": ActionProjector(action_dim=8),
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
                                    action_projector = [
                                        jnp.zeros((B, FLAGS.config.dataset_kwargs["action_chunk_size"], 8), dtype=jnp.float32),
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

    # Load from pre-trained Bagel checkpoint.
    # Orbax restore requires an exact pytree key match, so we strip the action_projector out first,
    # restore the remaining weights, then add the fresh action_projector params back.
    fresh_params = unfreeze(train_state.params)
    params_for_restore = {k: v for k, v in fresh_params.items() if k != "modules_action_projector"}
    restored_params = checkpointer.restore(FLAGS.config.pretrained_bagel_path, params_for_restore)
    restored_params = unfreeze(restored_params)  # ensure plain dict regardless of orbax version
    restored_params["modules_action_projector"] = fresh_params["modules_action_projector"]
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
            obs_dropout_prob=float(
                FLAGS.config.policy_kwargs.get("obs_dropout_prob", 0.0)
            ),
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
            obs_p = float(config["obs_dropout_prob"])
            # Prepare the vit tokens
            image = jnp.astype(batch["image"], jnp.float32) / 127.5 - 1
            if obs_p > 0:
                rng, rng_drop = jax.random.split(rng)
                B0 = batch["image"].shape[0]
                drop_obs = jax.random.bernoulli(rng_drop, obs_p, (B0,))
                image = jnp.where(
                    drop_obs[:, None, None, None],
                    jnp.asarray(0.0, dtype=jnp.float32),
                    image,
                )
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

            # Project action chunk to token embeddings
            action_tokens = train_state.apply_fn(
                {"params": params},
                action=batch["action/joint_velocity_chunk"],
                name="action_projector",
            )  # (B, T_action, hidden_dim)
            action_tokens = add_batch_sharding_constraint(action_tokens, where="action_tokens")

            # embed the text tokens
            text_embeds = train_state.apply_fn(
                {"params": params},
                token_ids=batch["text_tokens"],
                name="token_embedder",
            )
            text_embeds = add_batch_sharding_constraint(text_embeds, where="text_embeds")

            # Concat everything along sequence dimension: 
            # [img_tokens | action_token | text_tokens]
            full_seq = jnp.concatenate([pre_llm_vit_tokens, action_tokens, text_embeds], axis=1)
            full_seq = add_batch_sharding_constraint(full_seq, where="full_seq")

            # Prepare full seq rope IDs
            # Image tokens get rope_id=0 (since vit already has positional encoding)
            N_img = pre_llm_vit_tokens.shape[1]
            N_action = action_tokens.shape[1]
            image_rope_ids = jnp.zeros((N_img,), dtype=jnp.int32)[None, :]
            image_rope_ids = jnp.tile(image_rope_ids, (full_seq.shape[0], 1))

            action_rope_ids = (jnp.arange(N_action, dtype=jnp.int32) + 1)[None, :]
            action_rope_ids = jnp.tile(action_rope_ids, (full_seq.shape[0], 1))
            text_rope_ids = jnp.where(
                batch["text_token_masks"],
                batch["text_rope_ids"] + N_action,
                0,
            )
            full_seq_rope_ids = jnp.concatenate([image_rope_ids, action_rope_ids, text_rope_ids], axis=1)
            full_seq_rope_ids = add_batch_sharding_constraint(full_seq_rope_ids, where="full_seq_rope_ids")

            # Now prepare attention masks
            L = full_seq.shape[1]
            # Start with a causal mask
            causal = jnp.tril(jnp.ones((L, L), dtype=bool))
            # Enable full self-attention for vit tokens only (not action/text tokens).
            arr = jnp.concatenate([jnp.ones((N_img,), dtype=bool), jnp.zeros((L - N_img,), dtype=bool)])
            self_attention_mask = jnp.matmul(arr[:, None], arr[None, :])
            block_causal = causal | self_attention_mask
            # Now tile to include batch dimension
            block_causal = jnp.tile(block_causal[None, ...], (full_seq.shape[0], 1, 1))
            block_causal = add_batch_sharding_constraint(block_causal, where="block_causal")
            # padding sees nothing, and is seen by nothing
            # Text tokens can be padded. When obs dropout applies, image tokens are padded too.
            if obs_p > 0:
                img_padding = jnp.broadcast_to(
                    drop_obs[:, None], (full_seq.shape[0], N_img)
                )
            else:
                img_padding = jnp.zeros((full_seq.shape[0], N_img), dtype=bool)
            padding = jnp.concatenate([
                img_padding,
                jnp.zeros((full_seq.shape[0], N_action), dtype=bool),
                jnp.logical_not(batch["text_token_masks"]),
            ], axis=1)
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

            # Construct cross-entropy targets. DROID `distance` is timesteps-to-go; RoboArena `distance` is already a processed value in [0, 1] — use as-is.
            dist_f = batch["distance"].astype(jnp.float32)
            discounted_distances = jnp.where(
                batch["is_roboarena"],
                dist_f,
                jnp.power(config["discount_factor"], dist_f),
            )
            # Shape (B,), float32; values suitable for bucketing (typically in [0, 1]).
            assert discounted_distances.shape == (post_llm_seq.shape[0],)
            assert discounted_distances.dtype == jnp.float32
            bucket_ids = jnp.clip((discounted_distances * config["num_buckets"]).astype(jnp.int32), min=0, max=config["num_buckets"]-1) # values in [0, config["num_buckets"]-1]
            value_targets = config["num_buckets"] - bucket_ids - 1 # reverse so that closer distances have lower bucket ids
            value_targets = add_batch_sharding_constraint(value_targets, where="value_targets")

            # Critical: loss should be computed in float32
            value_logits = jnp.astype(value_logits, jnp.float32)

            per_example_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
                value_logits, value_targets
            )
            per_example_loss = add_batch_sharding_constraint(per_example_loss, where="loss before pmean")
            loss = jnp.mean(per_example_loss)

            # For logging, let's compute the accuracy in addition to the loss
            argmax_prediction_matches = jnp.argmax(value_logits, axis=-1) == value_targets
            accuracy = jnp.mean(argmax_prediction_matches)
            is_roboarena = batch["is_roboarena"]
            is_droid = jnp.logical_not(is_roboarena)

            def masked_mean(values, mask):
                mask_f = mask.astype(jnp.float32)
                denom = jnp.maximum(jnp.sum(mask_f), 1.0)
                return jnp.sum(values.astype(jnp.float32) * mask_f) / denom

            roboarena_loss = masked_mean(per_example_loss, is_roboarena)
            droid_loss = masked_mean(per_example_loss, is_droid)
            roboarena_accuracy = masked_mean(argmax_prediction_matches, is_roboarena)
            droid_accuracy = masked_mean(argmax_prediction_matches, is_droid)
            roboarena_count = jnp.sum(is_roboarena.astype(jnp.int32))
            droid_count = jnp.sum(is_droid.astype(jnp.int32))

            if obs_p > 0:
                obs_dropout_rate = jnp.mean(drop_obs.astype(jnp.float32))
            else:
                obs_dropout_rate = jnp.asarray(0.0, dtype=jnp.float32)

            log_dict = {
                "loss": loss,
                "accuracy": accuracy,
                "roboarena_loss": roboarena_loss,
                "droid_loss": droid_loss,
                "roboarena_accuracy": roboarena_accuracy,
                "droid_accuracy": droid_accuracy,
                "roboarena_count": roboarena_count,
                "droid_count": droid_count,
                "obs_dropout_rate": obs_dropout_rate,
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

    # Synchronize all workers before entering the training loop. Ensures no worker is ahead of others due to variable initialization time.
    multihost_utils.sync_global_devices("pre_train_loop")

    # Train loop
    timer = Timer()
    starting_train_step = int(jax.device_get(train_state.step))
    for i in tqdm.tqdm(range(starting_train_step, FLAGS.config.num_steps)):
        try:
            timer.tick("total")

            timer.tick("dataset")
            if use_roboarena:
                droid_batch = next(train_data_iter)
                robo_batch = next(roboarena_data_iter)
                batch = merge_batches(droid_batch, robo_batch)
                del droid_batch, robo_batch
            else:
                batch = next(train_data_iter)
                bsz = batch["image"].shape[0]
                batch["is_roboarena"] = np.zeros((bsz,), dtype=bool)
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
