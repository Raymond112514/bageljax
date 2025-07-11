from typing import Tuple, Dict, Any, Optional
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import dot_product_attention

class PatchEmbed(nn.Module):
    """14 × 14 conv-patchifier → (B, L, C)."""
    embed_dim: int = 1152
    patch: Tuple[int, int] = (14, 14)

    @nn.compact
    def __call__(self, x):
        x = (x.astype(jnp.float32) - 127.5) / 127.5   # [-1, 1]
        x = nn.Conv(self.embed_dim, self.patch,
                    strides=self.patch, padding='VALID',
                    use_bias=True, name="conv")(x)
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c), (h, w)            # token seq + grid size
    
class MHA(nn.Module):
    """Multi-head self-attention (flash/SDPA via `dot_product_attention`)."""
    heads: int = 16
    dropout_rate: float = 0.0          # keep 0.0 for inference

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,                 # (B, L, D)
                 *,
                 deterministic: bool = True,
                 mask: Optional[jnp.ndarray] = None):
        B, L, D = x.shape
        H       = self.heads
        d_h     = D // H                       # head dim (must divide evenly)

        dense = lambda name: nn.Dense(D, use_bias=True, name=name)

        # -------- QKV projections ----------------------------------------------------
        q = dense('q')(x).reshape(B, L, H, d_h)
        k = dense('k')(x).reshape(B, L, H, d_h)
        v = dense('v')(x).reshape(B, L, H, d_h)

        # -------- (optional) causal / padding mask ----------------------------------
        # `mask` should already be broadcastable to (B, H, L_q, L_k).
        y = dot_product_attention(
                query=q,
                key=k,
                value=v,
                mask=mask,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                precision='highest')           # enables flash/SDPA where available

        # -------- merge heads & output projection -----------------------------------
        y = y.reshape(B, L, D)
        return nn.Dense(D, use_bias=True, name='out')(y)

class MLP(nn.Module):
    projection_dim: int = 4304

    @nn.compact
    def __call__(self, x, *, deterministic=True):
        hidden_dim = x.shape[-1]
        x = nn.Dense(self.projection_dim, use_bias=True, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dense(hidden_dim, use_bias=True, name='fc2')(x)
        return x

class EncoderBlock(nn.Module):
    heads: int = 16
    projection_dim: int = 4304

    @nn.compact
    def __call__(self, x, *, deterministic=True):
        # attention
        h = nn.LayerNorm(name='ln1')(x)
        h = MHA(self.heads, name='mha')(h, deterministic=deterministic)
        x = x + h
        # MLP
        h = nn.LayerNorm(name='ln2')(x)
        h = MLP(self.projection_dim, name='mlp')(h, deterministic=deterministic)
        return x + h

class NaViT(nn.Module):
    depth: int = 26
    width: int = 1152
    heads: int = 16
    projection_dim: int = 4304
    max_grid: int = 70

    @nn.compact
    def __call__(self, img, *, deterministic=True):
        x, (hp, wp) = PatchEmbed(self.width, name='patch_embed')(img)  # (B,L,C)

        # internal positional embeddings (1152-d)
        pos = self.param('pre_vit_pos_embed',
                         nn.initializers.normal(0.02),
                         (self.max_grid ** 2, self.width))
        idx = jnp.arange(hp)[:, None] * self.max_grid + jnp.arange(wp)[None, :]
        x = x + jnp.take(pos, idx.reshape(-1), axis=0)

        # transformer encoder
        for i in range(self.depth):
            x = EncoderBlock(self.heads, self.projection_dim, name=f'block_{i}')(x, deterministic=deterministic)
        x = nn.LayerNorm(name='post_vit_ln')(x)
        return x, (hp, wp)

class ViTConnector(nn.Module):
    in_dim: int = 1152
    out_dim: int = 3584
    max_grid: int = 70

    @nn.compact
    def __call__(self, x, grid_hw):
        hp, wp = grid_hw
        # 2-layer GELU MLP
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc1')(x)
        h = nn.gelu(h)
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc2')(h)

        pos_xmodal = self.param('post_vit_pos_embed',
                                nn.initializers.normal(0.02),
                                (self.max_grid ** 2, self.out_dim))
        idx = jnp.arange(hp)[:, None] * self.max_grid + jnp.arange(wp)[None, :]
        return h + jnp.take(pos_xmodal, idx.reshape(-1), axis=0)

class BagelVisionEncoder(nn.Module):
    """NHWC pixels  → patch tokens in LLM space (3584-d)."""
    depth: int = 26

    @nn.compact
    def __call__(self, img, *, deterministic=True):
        vit_tokens, hw = NaViT(self.depth, name='vit')(img, deterministic=deterministic)
        llm_tokens = ViTConnector(name='connector')(vit_tokens, hw)
        return llm_tokens  # (B, L, 3584)