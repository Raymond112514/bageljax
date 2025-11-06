import os, json, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

class ResMLPBlock(nn.Module):
    hidden: int
    expand: int = 4

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.hidden * self.expand)(h)
        h = jax.nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        return x + h

class Encoder(nn.Module):
    hidden: int
    layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden)(x)
        for _ in range(self.layers):
            h = ResMLPBlock(self.hidden)(h)
        h = nn.LayerNorm()(h)
        z = nn.Dense(self.out_dim)(h)
        return jnp.tanh(z)

class ActionTokenizer(nn.Module):
    hidden: int = 768
    layers: int = 8

    def setup(self):
        self.radices = [32, 32]
        self.per_token_v = 32 * 32
        self.dims_per_token = len(self.radices)
        self.tokens_per_chunk = 8
        self.radix_mult = jnp.array([1, 32], dtype=jnp.int32)
        self.grids = [jnp.linspace(-1.0, 1.0, r, dtype=jnp.float32) for r in self.radices]
        self.flat_dim = 16 * 8

        self.enc = Encoder(self.hidden, self.layers, self.dims_per_token * self.tokens_per_chunk)

    def __call__(self, normalized_action_chunks):
        assert normalized_action_chunks.ndim == 3
        normalized_action_chunks = jnp.reshape(normalized_action_chunks, (normalized_action_chunks.shape[0], self.flat_dim))

        z = self.enc(normalized_action_chunks)

        def fsq_quantize(z_flat: jnp.ndarray):
            """
            z_flat: (B, self.dims_per_token * self.tokens_per_chunk) in [-1,1]
            Quantize per latent dim onto uniform grids per token-dimension, then pack to tokens.
            """
            B = z_flat.shape[0]
            z = z_flat.reshape(B, self.tokens_per_chunk, self.dims_per_token)  # (B,M,D)
            zq_list, idx_list = [], []
            for j in range(self.dims_per_token):
                grid = self.grids[j]                     # (Lj,)
                zj = z[:, :, j]                          # (B,M)
                d = jnp.abs(zj[..., None] - grid[None, None, :])
                idx = jnp.argmin(d, axis=-1).astype(jnp.int32)       # (B,M)
                qj = grid[idx]                           # (B,M)
                yj = zj + jax.lax.stop_gradient(qj - zj) # STE
                zq_list.append(yj)
                idx_list.append(idx)
            zq = jnp.stack(zq_list, axis=-1).reshape(B, self.tokens_per_chunk * self.dims_per_token)
            idxs = jnp.stack(idx_list, axis=-1)         # (B,M,D)
            tokens = jnp.sum(idxs * self.radix_mult[None, None, :], axis=-1)  # (B,M) in [0, self.per_token_v)
            return zq, tokens

        zq, tokens = fsq_quantize(z)

        return tokens

