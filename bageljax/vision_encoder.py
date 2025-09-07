from typing import Tuple, Dict, Any, Optional
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import dot_product_attention
from einops import rearrange

class PatchEmbed(nn.Module):
    """14 × 14 patchifier → (B, L, C)."""
    embed_dim: int = 1152
    patch_size: int = 14
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x): # we assume input images are already normalized to [-1, 1]
        x = x.astype(self.param_dtype) # cast to bfloat16 if not already
        x = rearrange(x, 'b (hn hp) (wn wp) c -> b hn hp wn wp c', hp=self.patch_size, wp=self.patch_size)
        x = rearrange(x, 'b hn hp wn wp c -> b hn wn (hp wp c)')
        _, h, w, _ = x.shape
        x = rearrange(x, 'b h w d -> b (h w) d')  # (B, L, C)
        x = nn.Dense(self.embed_dim, use_bias=True, name='proj', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        return x, (h, w)
    
class MHA(nn.Module):
    """Multi-head self-attention (flash/SDPA via `dot_product_attention`)."""
    heads: int = 16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,                 # (B, L, D)
                 *,
                 mask: Optional[jnp.ndarray] = None):
        B, L, D = x.shape
        H       = self.heads
        d_h     = D // H

        # ——— ensure activations arrive in bf16 ———
        x = x.astype(self.param_dtype)

        dense = lambda name: nn.Dense(D, use_bias=True, name=name, dtype=self.param_dtype, param_dtype=self.param_dtype)

        # -------- QKV projections ----------------------------------------------------
        q = dense('q')(x).reshape(B, L, H, d_h)
        k = dense('k')(x).reshape(B, L, H, d_h)
        v = dense('v')(x).reshape(B, L, H, d_h)

        # Attention. We set force_fp32_for_softmax=True following standard practice
        y = dot_product_attention(
                query=q,
                key=k,
                value=v,
                mask=mask,
                dropout_rate=0.0,
                deterministic=True,
                force_fp32_for_softmax=True,
        )
        # -------- merge heads & output projection -----------------------------------
        y = y.reshape(B, L, D)
        return nn.Dense(D, use_bias=True, name='out', dtype=self.param_dtype, param_dtype=self.param_dtype)(y)

class MLP(nn.Module):
    projection_dim: int = 4304
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.param_dtype)
        hidden_dim = x.shape[-1]
        x = nn.Dense(self.projection_dim, use_bias=True, name='fc1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(hidden_dim, use_bias=True, name='fc2', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        return x

class EncoderBlock(nn.Module):
    heads: int = 16
    projection_dim: int = 4304
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.param_dtype)

        # attention
        h = nn.LayerNorm(name='ln1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = MHA(self.heads, name='mha')(h)
        x = x + h
        # MLP
        h = nn.LayerNorm(name='ln2', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = MLP(self.projection_dim, name='mlp')(h)
        return x + h

class NaViT(nn.Module):
    depth: int = 26
    width: int = 1152
    heads: int = 16
    projection_dim: int = 4304
    max_grid: int = 70
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, img):
        x, (hp, wp) = PatchEmbed(self.width, name='patch_embed')(img)

        # internal positional embeddings (1152-d)
        pos = self.param('pre_vit_pos_embed',
                         nn.initializers.normal(0.02, dtype=self.param_dtype),
                         (self.max_grid ** 2, self.width))
        idx = jnp.arange(hp)[:, None] * self.max_grid + jnp.arange(wp)[None, :]
        x = x + jnp.take(pos, idx.reshape(-1), axis=0)

        # transformer encoder
        for i in range(self.depth):
            x = EncoderBlock(self.heads, self.projection_dim, name=f'block_{i}')(x)
        x = nn.LayerNorm(name='post_vit_ln', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        return x, (hp, wp)

class ViTConnector(nn.Module):
    in_dim: int = 1152
    out_dim: int = 3584
    max_grid: int = 70
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, grid_hw):
        hp, wp = grid_hw
        # 2-layer GELU MLP
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc2', dtype=self.param_dtype, param_dtype=self.param_dtype)(h)

        pos_xmodal = self.param('post_vit_pos_embed',
                                nn.initializers.normal(0.02, dtype=self.param_dtype),
                                (self.max_grid ** 2, self.out_dim))
        idx = jnp.arange(hp)[:, None] * self.max_grid + jnp.arange(wp)[None, :]
        return h + jnp.take(pos_xmodal, idx.reshape(-1), axis=0)

class VisionEncoder(nn.Module):
    """NHWC pixels  → patch tokens in LLM space (3584-d)."""
    depth: int = 26

    @nn.compact
    def __call__(self, img):
        vit_tokens, hw = NaViT(self.depth, name='vit')(img)
        llm_tokens = ViTConnector(name='connector')(vit_tokens, hw)
        return llm_tokens  # (B, L, 3584)