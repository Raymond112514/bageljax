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
from flax.training.train_state import TrainState as FlaxTrainState
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import threading
from etils import epath
from flax.training import orbax_utils

from bageljax.common.wandb import WandBLogger
from bageljax.common.common import TrainState, ModuleDict, nonpytree_field
from bageljax.common.optimizers import make_optimizer
from bageljax.common.typing import Batch, PRNGKey
from bageljax.data.dataset import glob_to_path_list, Dataset
from bageljax.utils.timer_utils import Timer
from bageljax.utils.jax_utils import host_broadcast_str, create_sharding, gather_train_state, add_batch_sharding_constraint, enforce_sharding_constraints, unset_context_mesh, reset_context_mesh
from bageljax.model.vocabulary import TokenEmbedder, LogitsHead
from bageljax.model.vision_encoder import VisionEncoder
from bageljax.model.mixture_of_transformers import MixtureOfTransformers
from bageljax.model.tokenizer import Qwen2Tokenizer, add_special_tokens
from bageljax.model.proprio_projector import ProprioProjector
from bageljax.model.action_tokenizer import ActionTokenizer

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

def main(_):
    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": f"worldmodelrl",
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
    dataset_tfrecords = glob_to_path_list("*.tfrecord*", FLAGS.config.dataset_kwargs["dataset_path"]) # returns a list of paths

    # Use different dataset seeds for each worker
    dataset_seed = FLAGS.config.seed + jax.process_index()

    # Create the training dataset
    train_data = Dataset(
        data_paths=dataset_tfrecords,
        seed=dataset_seed,
        action_proprio_metadata=FLAGS.config.dataset_kwargs["action_proprio_metadata"],
        batch_size=FLAGS.config.dataset_kwargs["batch_size"],
        shuffle_buffer_size=FLAGS.config.dataset_kwargs["shuffle_buffer_size"],
        chunk_size=FLAGS.config.dataset_kwargs["chunk_size"],
        num_parallel_calls=FLAGS.config.dataset_kwargs["num_parallel_calls"],
        train=True,
    )
    train_data_iter = train_data.iterator()
    example_batch = next(train_data_iter)
    print("Data loader initialized")

    # Create the language tokenizer
    tokenizer_load_path = FLAGS.config.tokenizer_load_path
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Function to convert text instruction strings into padded tokens, and adds to the batch 
    # these tokens as well as the token masks and text rope IDs
    def tokenize_and_pad(batch):
        # What this function is going to do is rewrite the language instruction into:
        # "What actions should the robot take to complete the following language instruction:\n\n<language_instruction>\n\nActions:"
        # and then tokenize. It will also left-pad the text to the max length, and assign RoPE IDs.

        PAD_TOKEN_ID = 0 # it doesn't really matter what this is
        MAX_PROMPT_LENGTH = FLAGS.config.dataset_kwargs["max_prompt_length"] + 20 # approx 20 tokens for the prompt filler text

        B = batch["image"].shape[0]
        batch_tokenized_language = []
        batch_masks = []
        batch_text_rope_ids = []
        for i in range(B):
            rope_ids = []
            prompt = "What actions should the robot take to complete the following language instruction:\n\n"
            prompt += batch["language_instruction"][i].decode("utf-8").strip()
            prompt += "\n\nActions:"
            prompt_text_ids = tokenizer.encode(prompt)
            
            # Add bos and eos tokens
            prompt_text_ids = [new_token_ids['bos_token_id']] + prompt_text_ids + [new_token_ids['eos_token_id']]

            # Also add the bos token at the end to start action token generation
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
                np.arange(non_pad_tokens_in_prompt, dtype=np.int32) + 2, # start from 2, since the image and proprio come before
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

    # Create the FSQ encoder
    print("Initializing FSQ action tokenizer...")
    action_tokenizer = ActionTokenizer()
    
    def action_tokenizer_init_fn(rng):
        rng, init_rng = jax.random.split(rng)
        params = action_tokenizer.init({'params': init_rng}, jnp.zeros((1, FLAGS.config.dataset_kwargs["chunk_size"], 8), dtype=jnp.float32))["params"]
        rng, create_rng = jax.random.split(rng)
        state = TrainState.create(
            apply_fn=action_tokenizer.apply,
            params=params,
            txs=None,
            target_params=None,
            rng=create_rng,
            force_f32=True, # this model doesn't use mixed precision
        )
        return state

    rng, key = jax.random.split(rng)
    action_tokenizer_train_state_shape = jax.eval_shape(action_tokenizer_init_fn, key)

    # Create sharding and train_state
    data_sharding, action_tokenizer_train_state_sharding, no_shard, shard_data, global_to_local = create_sharding("fsdp", action_tokenizer_train_state_shape)
    rng, key = jax.random.split(rng)
    action_tokenizer_train_state = jax.jit(action_tokenizer_init_fn, out_shardings=action_tokenizer_train_state_sharding)(key)

    # Load action tokenizer from checkpoint
    # The action tokenizer checkpoint wasn't saved as a train state, but rather just the param pytree
    # This checkpoint loading process will also include checks to ensure that the params change (meaning the checkpoint is indeed loaded correctly)
    #loaded_params = checkpoints.restore_checkpoint(
    #    FLAGS.config.action_tokenizer_resume_path,
    #    target=action_tokenizer_train_state.params,
    #)
    #from flax.traverse_util import flatten_dict
    #def _flat(d): return {"/".join(k): v for k, v in flatten_dict(d, sep="/").items()}
    #f_tgt, f_ld = _flat(action_tokenizer_train_state.params), _flat(loaded_params)
    #missing = [k for k in f_tgt if k not in f_ld]
    #extra   = [k for k in f_ld if k not in f_tgt]
    #shape_mismatch = [(k, f_ld[k].shape, f_tgt[k].shape)
    #                for k in f_ld.keys() & f_tgt.keys()
    #                if f_ld[k].shape != f_tgt[k].shape]
    #if missing or extra or shape_mismatch:
    #    raise ValueError(f"CKPT mismatch. missing={missing[:5]}, extra={extra[:5]}, shape_mismatch={shape_mismatch[:5]}")
    #action_tokenizer_train_state = action_tokenizer_train_state.replace(params=loaded_params)

    # Print number of params of action tokenizer
    print("Total action tokenizer parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(action_tokenizer_train_state.params)]))

    # Function to replace the action chunks in an assumed sharded batch with action tokens, applying the action tokenizer
    @partial(
        jax.jit,
        in_shardings=(action_tokenizer_train_state_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    def tokenize_action_chunks(action_tokenizer_train_state, batch):
        action_tokens = action_tokenizer_train_state.apply_fn(
            {"params": action_tokenizer_train_state.params},
            normalized_action_chunks=batch["action_chunks"],
        )
        assert action_tokens.ndim == 2
        assert action_tokens.dtype == jnp.int32

        # Add in the vocabulary offset
        action_tokens = action_tokens + FLAGS.config.dataset_kwargs["action_tokens_offset"]
        action_tokens = add_batch_sharding_constraint(action_tokens, where="action tokenizer output")

        del batch["action_chunks"]
        batch["action_tokens"] = action_tokens

        return batch

    # Initialize the main model
    print("Initializing BagelVLA model...")

    networks = {
        "token_embedder": TokenEmbedder(),
        "vision_encoder": VisionEncoder(),
        "mixture_of_transformers": MixtureOfTransformers(),
        "logits_head": LogitsHead(),
        "proprio_projector": ProprioProjector(),
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
                                    proprio_projector = [
                                        jnp.zeros((B, 7), dtype=jnp.bfloat16), # remember, proprio is just 7 dimensions because we're not feeding in the gripper dimension
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

    # Load from pre-trained Bagel checkpoint
    # Checkpoint loading requires some extra logic here
    #print("Loading from pre-trained Bagel checkpoint...")
    # First delete the proprio projector from the pytree (legacy flax checkpoints doesn't allow the target to have keys the checkpoint doesn't)
    #proprio_projector_params = train_state.params["modules_proprio_projector"]
    #del train_state.params["modules_proprio_projector"]
    # Next construct a Flax TrainState
    #flax_train_state = FlaxTrainState.create(
    #    apply_fn=train_state.apply_fn,
    #    params=train_state.params,
    #    tx=optax.identity(),
    #)
    #flax_train_state = checkpoints.restore_checkpoint(
    #    FLAGS.config.pretrained_bagel_path,
    #    target=flax_train_state,
    #)
    #flax_train_state.params["modules_proprio_projector"] = proprio_projector_params
    #train_state = train_state.replace(params=flax_train_state.params)
    # At this point, everything has been restored correctly, except for the train_state's target parameters
    # This is a (kinda hacky) fix to properly re-initialize the target params
    #train_state = train_state.target_update(tau=1.0)
    #print("Loaded from pre-trained Bagel")

    # Load from a previous checkpoint if necessary
    if FLAGS.config.get("resume_path", None) is not None:
        print("Loading from previous checkpoint...")
        train_state = checkpoints.restore_checkpoint(
            FLAGS.config.resume_path,
            target=train_state,
        )
        print("Loaded from previous checkpoint")

    # Print total number of parameters of the model
    print("Total model parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(train_state.params)]))

    # Create auxiliary objects for use by update function
    config = flax.core.FrozenDict(
        dict(
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
            # Let's start with embedding proprio into the LLM space
            pre_llm_proprio_token = train_state.apply_fn(
                {"params": params},
                proprio=batch["proprio"],
                name="proprio_projector",
            )
            pre_llm_proprio_token = pre_llm_proprio_token[:, None, :] # add a sequence dimension
            pre_llm_proprio_token = add_batch_sharding_constraint(pre_llm_proprio_token, where="pre_llm_proprio_token")

            # Next the image
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

            # Embed the action tokens, dropping the last token (it will only be used as a target)
            action_tokens = batch["action_tokens"][:, :-1]
            action_token_embeds = train_state.apply_fn(
                {"params": params},
                token_ids=action_tokens,
                name="token_embedder",
            )
            action_token_embeds = add_batch_sharding_constraint(action_token_embeds, where="action_token_embeds")

            # Concat everything along sequence dimension
            full_seq = jnp.concatenate([pre_llm_proprio_token, pre_llm_vit_tokens, text_embeds, action_token_embeds], axis=1)
            full_seq = add_batch_sharding_constraint(full_seq, where="full_seq")

            # Prepare full seq rope IDs
            proprio_and_image_rope_ids = jnp.concatenate([
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=jnp.int32),
            ])[None, :]
            proprio_and_image_rope_ids = jnp.tile(proprio_and_image_rope_ids, (full_seq.shape[0], 1))

            proprio_image_and_text_rope_ids = jnp.concatenate([proprio_and_image_rope_ids, batch["text_rope_ids"]], axis=1)
            action_rope_ids = jnp.tile((jnp.arange(action_tokens.shape[1], dtype=jnp.int32) + 1)[None, :], (full_seq.shape[0], 1))
            action_rope_ids = action_rope_ids + proprio_image_and_text_rope_ids[:, -1:]
            full_seq_rope_ids = jnp.concatenate([proprio_image_and_text_rope_ids, action_rope_ids], axis=1)
            full_seq_rope_ids = add_batch_sharding_constraint(full_seq_rope_ids, where="full_seq_rope_ids")

            # Now prepare attention masks
            # To maintain closeness with pretraining, image and text tokens will not see proprio, even though it comes 
            # before in the sequence. However action tokens will, starting from the bos token
            L = full_seq.shape[1]
            # Start with a causal mask
            causal = jnp.tril(jnp.ones((L, L), dtype=bool))
            # the first token (proprio) can see itself, but all other tokens, up to the set of action tokens, cannot
            replacement_first_col = jnp.concatenate([jnp.ones((1,), dtype=bool), jnp.zeros((L-2-action_token_embeds.shape[1],), dtype=bool), jnp.ones((action_token_embeds.shape[1]+1,), dtype=bool)])
            causal = jnp.concatenate([replacement_first_col[:, None], causal[:, 1:]], axis=1)
            # Enable full self-attention for vit tokens
            arr = jnp.concatenate([jnp.zeros((1,), dtype=bool), jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=bool), jnp.zeros((L - pre_llm_vit_tokens.shape[1] - 1), dtype=bool)])
            self_attention_mask = jnp.matmul(arr[:, None], arr[None, :])
            block_causal = causal | self_attention_mask
            # Now tile to include batch dimension
            block_causal = jnp.tile(block_causal[None, ...], (full_seq.shape[0], 1, 1))
            # padding sees nothing, and is seen by nothing
            padding = jnp.concatenate([jnp.zeros((full_seq.shape[0], 1 + pre_llm_vit_tokens.shape[1]), dtype=bool), jnp.logical_not(batch["text_token_masks"]), jnp.zeros((full_seq.shape[0], action_token_embeds.shape[1]), dtype=bool)], axis=1)
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

            # Extract just the action tokens
            post_llm_action_tokens = post_llm_seq[:, -batch["action_tokens"].shape[1]:]

            # Feed into LLM head
            action_prediction_logits = train_state.apply_fn(
                {"params": params},
                hidden_states=post_llm_action_tokens,
                name="logits_head",
            )
            # this should have shape (B, num_action_tokens, vocab_dim)
            action_prediction_logits = add_batch_sharding_constraint(action_prediction_logits, where="action_prediction_logits")

            # Construct cross-entropy targets
            action_token_targets = batch["action_tokens"] # just this, no change needed

            # Critical: loss should be computed in float32
            action_prediction_logits = jnp.astype(action_prediction_logits, jnp.float32)

            loss = optax.losses.softmax_cross_entropy_with_integer_labels(action_prediction_logits, action_token_targets)
            loss = add_batch_sharding_constraint(loss, where="loss before pmean")
            loss = jnp.mean(loss)

            # For logging, let's compute the per-token classification accuracy, and all-token accuracy
            argmax_prediction_matches = jnp.argmax(action_prediction_logits, axis=-1) == action_token_targets
            token_accuracy = jnp.mean(argmax_prediction_matches)
            per_token_accuracies = jnp.mean(argmax_prediction_matches, axis=0)
            seq_accuracy = jnp.mean(jnp.prod(argmax_prediction_matches, axis=-1))

            log_dict = {
                "loss": loss,
                "token_accuracy": token_accuracy,
                "seq_accuracy": seq_accuracy,
            }
            for token_num in range(per_token_accuracies.shape[0]):
                log_dict[f"token{token_num}_acc"] = per_token_accuracies[token_num]

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

    # Build one checkpointer per process (top-level, not per-step).
    _checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    # Optional: a simple lock so you don’t overlap saves on one host.
    _ckpt_lock = threading.Lock()

    def _barrier_mesh():
        """Returns the (processes, local_devices) mesh that JAX multihost utils expect."""
        devs = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())
        return jax.sharding.Mesh(devs, ('processes', 'local_devices'))

    def save_ckpt(save_dir: str, step: int, state):
        """Call on ALL hosts. Orbax will write each host's shards into the same step dir."""
        step_dir = epath.Path(save_dir) / f"{int(step):08d}"
        save_args = orbax_utils.save_args_from_target(state)  # build metadata once

        def _worker():
            # IMPORTANT: use the barrier mesh while calling .save() to avoid device-id mismatch.
            with _barrier_mesh():
                # AsyncCheckpointer.save returns immediately; Orbax writes in the background.
                _checkpointer.save(step_dir, args=ocp.args.StandardSave(state, save_args=save_args))

        # Launch on a clean thread to decouple from any training pjit contexts.
        with _ckpt_lock:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()

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
            batch = tokenize_action_chunks(action_tokenizer_train_state, batch)
            timer.tock("dataset")

            timer.tick("train")
            train_state, update_info = update(train_state, batch, config, lr_schedule)
            timer.tock("train")

            #def _block(x):
            #    if isinstance(x, jax.Array):
            #        x.block_until_ready()
            #    return None

            #jax.tree_util.tree_map(_block, (train_state, update_info))

            timer.tock("total")

            step = i + 1

            if step % FLAGS.config.save_interval == 0 or step == 20:
                save_ckpt(save_dir, step, train_state)

            if step % FLAGS.config.log_interval == 0:
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
