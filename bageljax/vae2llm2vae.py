from pathlib import Path
from typing import Tuple

import einops
import jax, jax.numpy as jnp
import flax.linen as nn


def group_2x2(z16: jnp.ndarray) -> jnp.ndarray:
    """(B, H, W, 16) → (B, H/2, W/2, 64)."""
    b, h, w, c = z16.shape
    z = z16.reshape(b, h // 2, 2, w // 2, 2, c)    
    z = z.transpose(0, 1, 3, 2, 4, 5)               
    return z.reshape(b, h // 2, w // 2, 4 * c)       


def ungroup_2x2(z64: jnp.ndarray) -> jnp.ndarray:
    """(B, H, W, 64) → (B, H*2, W*2, 16)."""
    b, h, w, c = z64.shape
    z = z64.reshape(b, h, w, 2, 2, c // 4)
    z = z.transpose(0, 1, 3, 2, 4, 5)
    return z.reshape(b, h * 2, w * 2, c // 4)


def timestep_embedding(
    timesteps: jnp.ndarray,
    dim: int = 256,
    max_period: int = 10_000,
) -> jnp.ndarray:
    """
    Sinusoidal timestep embedding. This happens in float32, and later the embeddings returned
    will be projected into the LLM dimension, an operation that will happen in bfloat16.

    Args:
        timesteps: 1-D array of shape (N,) with float timesteps.
        dim:       Embedding dimension.
        max_period: Controls the minimum frequency.

    Returns:
        (N, dim) array of float32 embeddings (cosine part first, then sine).
    """
    # --- all float32 from here on ------------------------------------------
    t      = timesteps.astype(jnp.float32)           # (N,)
    half   = dim // 2
    half_f = jnp.float32(half)                      # scalar float32

    freqs = jnp.exp(
        -jnp.log(jnp.float32(max_period))           # scalar float32
        * jnp.arange(half, dtype=jnp.float32)       # (half,)
        / half_f
    )                                               # (half,)

    args = t[:, None] * freqs[None, :]              # (N, half)
    emb  = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return emb.astype(jnp.float32)                  # (N, dim) float32


class TimeEmbedder(nn.Module):
    llm_dim: int = 3584
    emb_dim: int = 256
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.dense0 = nn.Dense(self.llm_dim,
                               use_bias=True,
                               name="mlp/dense0",
                               dtype=self.param_dtype,
                               param_dtype=self.param_dtype)
        self.dense1 = nn.Dense(self.llm_dim,
                               use_bias=True,
                               name="mlp/dense1",
                               dtype=self.param_dtype,
                               param_dtype=self.param_dtype)

    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        emb = timestep_embedding(timesteps, self.emb_dim)
        emb = emb.astype(self.param_dtype)  # timestep_embedding func returns float32
        h = nn.swish(self.dense0(emb))
        return self.dense1(h)


class VAE2LLM(nn.Module):
    """Produces (B, L, 3584) tokens from VAE latents."""
    max_grid: int = 64
    llm_dim : int = 3584
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.up = nn.Dense(self.llm_dim,
                           use_bias=True,
                           name="vae2llm",
                           dtype=self.param_dtype,
                           param_dtype=self.param_dtype)
        self.pos_embed = self.param("pos_embed",
                                    nn.initializers.normal(0.02, dtype=self.param_dtype),
                                    (self.max_grid**2, self.llm_dim))

    def __call__(self,
                 z16: jnp.ndarray,
                 ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        """Returns tokens and latent grid size (h,w)."""
        z64 = group_2x2(z16)                               # (B,H/16,W/16,64)

        b, h, w, _ = z64.shape
        tokens64 = einops.rearrange(z64, "b h w c -> b (h w) c")  # flatten
        hid = self.up(tokens64)                                    # 64 → 3584

        # add absolute pos-embed
        idx = (jnp.arange(h)[:, None] * self.max_grid + jnp.arange(w)[None, :]).reshape(-1)
        hid = hid + jnp.take(self.pos_embed, idx, axis=0)
        return hid, (h, w)                                         # (B,L,3584)


class LLM2VAE(nn.Module):
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.down = nn.Dense(64, use_bias=True, name="llm2vae", dtype=self.param_dtype, param_dtype=self.param_dtype)  # 3584 → 64

    def __call__(self,
                 tokens: jnp.ndarray,
                 grid_hw: Tuple[int, int],
                 ) -> jnp.ndarray:
        """
        tokens : (B, L, 3584)  – from BAGEL image-expert
        grid_hw: latent grid (h, w) used during encoding
        returns VAE latents
        """
        tokens = tokens.astype(self.param_dtype)  # ensure correct dtype

        b, l, _ = tokens.shape
        h, w    = grid_hw
        assert h * w == l, "grid size does not match token count"

        z64 = self.down(tokens)                               # (B,L,64)
        z64 = z64.reshape(b, h, w, 64)                        # grid
        z16 = ungroup_2x2(z64)                                # (B,H*2,W*2,16)
        return z16.astype(jnp.float32)                        # VAE portion of model operates in float32
