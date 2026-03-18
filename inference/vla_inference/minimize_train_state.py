import jax
jax.distributed.initialize()

import os
import tensorflow as tf
from functools import partial
import jax.numpy as jnp
import optax
import numpy as np
import tqdm
import traceback
import random
import copy
import datetime
from einops import rearrange
import flax
from flax.core import freeze, unfreeze, FrozenDict
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from etils import epath
from typing import Dict
from PIL import Image
import asyncio
import dataclasses
import logging
import time
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from bageljax.common.common import TrainState, ModuleDict, nonpytree_field
from bageljax.common.optimizers import make_optimizer
from bageljax.utils.jax_utils import create_sharding, add_batch_sharding_constraint, enforce_sharding_constraints
from bageljax.model.vocabulary import TokenEmbedder, LogitsHead
from bageljax.model.vision_encoder import VisionEncoder
from bageljax.model.mixture_of_transformers import MixtureOfTransformers

def get_value_function_last_number(path: str) -> str:
    return path.rstrip('/').split('/')[-1]

# --------------- all configs/hyperparams for inference are stored here, modify at will ---------------
INFERENCE_CONFIG = {
    "seed": 0,
    "checkpoint_load_dir": "gs://pranav-us-west1/log/worldmodelrl/value_function_20251211_233210/00100000", 
    "reduced_checkpoint_save_dir": "gs://raymond-us-west1/value_function",
}
INFERENCE_CONFIG["reduced_checkpoint_save_dir"] = f"{INFERENCE_CONFIG['reduced_checkpoint_save_dir']}/{get_value_function_last_number(INFERENCE_CONFIG['checkpoint_load_dir'])}"

# Initialize rng from config seed
rng = jax.random.PRNGKey(INFERENCE_CONFIG["seed"])

# We're starting model creation; set enforce sharding constraints to false for this phase
enforce_sharding_constraints(False)

# Create checkpointer object (used for all subsequent checkpoint loading)
checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())

# Initialize the main model (value function)
print("Initializing Bagel Value Function model...")

networks = {
    "token_embedder": TokenEmbedder(),
    "vision_encoder": VisionEncoder(),
    "mixture_of_transformers": MixtureOfTransformers(),
    "logits_head": LogitsHead(),
}

model_def = ModuleDict(networks)

# create optimizer just for train state initialization
tx, lr_schedule = make_optimizer(
    learning_rate=0.01,
    weight_decay=0.01,
    beta2=0.5,
    eps=1e-6,
    cosine_decay_steps=None,
    warmup_steps=2000,
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

# Load from checkpoint
print("Loading from previous checkpoint...")
train_state = checkpointer.restore(
    INFERENCE_CONFIG["checkpoint_load_dir"],
    train_state,
)
print("Loaded from previous checkpoint")

# Convert target_params to bfloat16 to reduce checkpoint size
train_state = train_state.make_target_params_bfloat16()
print(f"Done converting to bfloat16")

# Prepare for checkpointing the model
checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

def save_ckpt(save_dir: str, obj):
    save_dir = epath.Path(save_dir)
    save_args = orbax_utils.save_args_from_target(obj)
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=('data',))
    with mesh:
        print(f"Worker {jax.process_index()} starting async save...")
        checkpointer.save(
            save_dir,
            args=ocp.args.StandardSave(obj, save_args=save_args)
        )
        checkpointer.wait_until_finished()

# Execute the save
save_ckpt(INFERENCE_CONFIG["reduced_checkpoint_save_dir"], train_state.target_params)
print("Save complete and verified on GCS.")
