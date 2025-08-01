import os
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from typing import Any, Callable, Optional, Tuple
from flax.training import checkpoints
from flax.training.train_state import TrainState
import numpy as np
import optax
import torch
from safetensors.torch import load_file as load_sft
from copy import deepcopy

from bageljax.vocabulary import TokenEmbedder, LogitsHead
from bageljax.vision_encoder import VisionEncoder
from bageljax.vae2llm2vae import TimeEmbedder, VAE2LLM, LLM2VAE
from bageljax.mixture_of_transformers import MixtureOfTransformers
from bageljax.common import ModuleDict

# We will be saving checkpoints in this script, and we want to avoid using orbax
# Prevent flax.checkpoints from using Orbax backend
flax.config.update('flax_use_orbax_checkpointing', False)

# Create the model
networks = {
    "token_embedder": TokenEmbedder(),
    "vision_encoder": VisionEncoder(),
    "time_embedder": TimeEmbedder(),
    "vae2llm": VAE2LLM(),
    "llm2vae": LLM2VAE(),
    "mixture_of_transformers": MixtureOfTransformers(),
    "logits_head": LogitsHead(),
}

model_def = ModuleDict(networks)

def init_fn(rng):
    rng, init_rng = jax.random.split(rng)

    # For init, let's pick reasonable values of some of the input parameters
    B = 1 # batch size
    H, W = 672, 672 # image height and width, 672 is divisible by 14 and 16
    vae_latent_dim = 16 # VAE latent dimension
    llm_hidden_dim = 3584 # LLM hidden dimension
    L = 42*42 # sequence length, for llm2vae needs to be (H/16)*(W/16)

    params = model_def.init({'params': init_rng},
                                token_embedder = [
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                ],
                                vision_encoder = [
                                    jnp.zeros((B, H, W, 3), dtype=jnp.uint8),
                                ],
                                time_embedder = [
                                    jnp.zeros((B,), dtype=jnp.float32),
                                ],
                                vae2llm = [
                                    jnp.zeros((B, H // 8, W // 8, vae_latent_dim), dtype=jnp.bfloat16),
                                ],
                                llm2vae = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                    (H // 16, W // 16),
                                ],
                                mixture_of_transformers = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                    jnp.zeros((B, 1, L, L), dtype=jnp.bfloat16),
                                ],
                                logits_head = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                ],
                            )["params"]
    train_state = TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        tx=optax.identity(),
    )
    return train_state

rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
train_state = jax.jit(init_fn)(key)

# Load the checkpoint
checkpoint_path = ...