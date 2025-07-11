# vae2llm2vae.py
# ---------------------------------------------------------------------
#  Bridging logic between Flux-style VAE (stride-8, 16-ch) and BAGEL LLM
# ---------------------------------------------------------------------
from pathlib import Path
from typing   import Tuple

import einops
import jax, jax.numpy as jnp
import flax.linen as nn


def group_2x2(z16: jnp.ndarray) -> jnp.ndarray:
    """(B, H, W, 16) → (B, H/2, W/2, 64)."""
    b, h, w, c = z16.shape
    z = z16.reshape(b, h // 2, 2, w // 2, 2, c)          # split spatial
    z = z.transpose(0, 1, 3, 2, 4, 5)                    # move tiles next
    return z.reshape(b, h // 2, w // 2, 4 * c)           # merge => 64


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
    Sinusoidal timestep embedding (float32-only).

    Args:
        timesteps: 1-D array of shape (N,) with integer or float timesteps.
        dim:       Embedding dimension (even is recommended).
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


# ---------------------------------------------------------------------
#  1)  latent tokens → LLM hidden
# ---------------------------------------------------------------------
class VAE2LLM(nn.Module):
    """Produces (B, L, 3584) tokens from VAE latents."""
    max_grid: int = 64          # 64 × 64 = 4096
    llm_dim : int = 3584

    # --- parameters ------------------------------------------------------------------
    def setup(self):
        # learned 2-layer up-projection   64 → 3584
        self.up = nn.Dense(self.llm_dim,
                           use_bias=True,
                           name="vae2llm")               # (W,b)
        # absolute position table for 64×64 grid
        self.pos_embed = self.param("pos_embed",
                                    nn.initializers.normal(0.02),
                                    (self.max_grid**2, self.llm_dim))

    # --- forward ----------------------------------------------------------------------
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


# ---------------------------------------------------------------------
#  2)  Sinusoidal timestep → hidden-dim conditioning
# ---------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    llm_dim: int = 3584
    emb_dim: int = 256

    def setup(self):
        self.dense0 = nn.Dense(self.llm_dim,
                               use_bias=True,
                               name="mlp/dense0")          # (3584,256)
        self.dense1 = nn.Dense(self.llm_dim,
                               use_bias=True,
                               name="mlp/dense1")          # (3584,3584)

    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        emb = timestep_embedding(timesteps, self.emb_dim)        # (B,256)
        h   = nn.swish(self.dense0(emb))
        return self.dense1(h)                                    # (B,3584)


# ---------------------------------------------------------------------
#  3)  LLM hidden → latent grid → decoded RGB
# ---------------------------------------------------------------------
class LLM2VAE(nn.Module):
    max_grid: int = 64          # for reshaping sanity-check
    llm_dim : int = 3584

    def setup(self):
        self.down = nn.Dense(64, use_bias=True, name="llm2vae")  # 3584 → 64

    def __call__(self,
                 tokens: jnp.ndarray,
                 grid_hw: Tuple[int, int],
                 ) -> jnp.ndarray:
        """
        tokens : (B, L, 3584)  – from BAGEL image-expert
        grid_hw: latent grid (h, w) used during encoding
        returns VAE latents
        """
        b, l, _ = tokens.shape
        h, w    = grid_hw
        assert h * w == l, "grid size does not match token count"

        z64 = self.down(tokens)                               # (B,L,64)
        z64 = z64.reshape(b, h, w, 64)                        # grid
        z16 = ungroup_2x2(z64)                                # (B,H*2,W*2,16)
        return z16
