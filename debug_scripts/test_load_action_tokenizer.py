import tensorflow as tf
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils, mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental.multihost_utils import process_allgather
import flax
import optax
import numpy as np
import tqdm
import traceback
import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import os
import random
import copy
import datetime
from einops import rearrange

from bageljax.common.common import TrainState, ModuleDict, nonpytree_field
from bageljax.common.typing import Batch, PRNGKey
from bageljax.model.action_tokenizer import ActionTokenizer

# Prevent flax from using the orbax backend for checkpoints
flax.config.update('flax_use_orbax_checkpointing', False)

# Initialize rng
rng = jax.random.PRNGKey(0)

# Create the FSQ encoder
action_tokenizer = ActionTokenizer()

def action_tokenizer_init_fn(rng):
    rng, init_rng = jax.random.split(rng)
    params = action_tokenizer.init({'params': init_rng}, jnp.zeros((1, 16, 8), dtype=jnp.float32))["params"]
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

# Create train_state
rng, key = jax.random.split(rng)
action_tokenizer_train_state = jax.jit(action_tokenizer_init_fn)(key)

# Load action tokenizer from checkpoint
# The action tokenizer checkpoint wasn't saved as a train state, but rather just the param pytree
# This checkpoint loading process will also include checks to ensure that the params change (meaning the checkpoint is indeed loaded correctly)
loaded_params = checkpoints.restore_checkpoint(
    "/home/pranav/bageljax/action_tokenizer_ckpt/checkpoint_10",
    target=action_tokenizer_train_state.params,
)

from flax.traverse_util import flatten_dict
def _flat(d): return {"/".join(k): v for k, v in flatten_dict(d, sep="/").items()}
f_tgt, f_ld = _flat(action_tokenizer_train_state.params), _flat(loaded_params)
missing = [k for k in f_tgt if k not in f_ld]
extra   = [k for k in f_ld if k not in f_tgt]
shape_mismatch = [(k, f_ld[k].shape, f_tgt[k].shape)
                for k in f_ld.keys() & f_tgt.keys()
                if f_ld[k].shape != f_tgt[k].shape]
if missing or extra or shape_mismatch:
    raise ValueError(f"CKPT mismatch. missing={missing[:5]}, extra={extra[:5]}, shape_mismatch={shape_mismatch[:5]}")
action_tokenizer_train_state = action_tokenizer_train_state.replace(params=loaded_params)

# Print number of params of action tokenizer
print("Total action tokenizer parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(action_tokenizer_train_state.params)]))

# Function to replace the action chunks in an assumed sharded batch with action tokens, applying the action tokenizer
@jax.jit
def tokenize_action_chunks(action_tokenizer_train_state, batch):
    action_tokens = action_tokenizer_train_state.apply_fn(
        {"params": action_tokenizer_train_state.params},
        normalized_action_chunks=batch["action_chunks"],
    )
    assert action_tokens.ndim == 2
    assert action_tokens.dtype == jnp.int32

    # Add in the vocabulary offset
    action_tokens = action_tokens + 100

    del batch["action_chunks"]
    batch["action_tokens"] = action_tokens

    return batch

batch = {
    "action_chunks": jnp.zeros((10, 16, 8), dtype=jnp.float32),
}
batch = tokenize_action_chunks(action_tokenizer_train_state, batch)
print("Result of tokenization...")
print(batch.keys())
print(batch["action_tokens"].shape, batch["action_tokens"].dtype)
print(batch["action_tokens"][0])