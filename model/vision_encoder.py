from typing import Tuple, Dict, Any
import jax.numpy as jnp
import flax.linen as nn

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
    heads: int = 16

    @nn.compact
    def __call__(self, x, *, deterministic=True):
        hidden_dim = x.shape[-1]
        proj = lambda n: nn.Dense(hidden_dim, use_bias=True, name=n)(x)
        q, k, v = (proj(n) for n in ['q', 'k', 'v'])
        b, l, d = q.shape; h = self.heads; d_h = d // h
        split = lambda t: t.reshape(b, l, h, d_h).transpose(0, 2, 1, 3)
        q, k, v = map(split, (q, k, v))
        attn = jnp.einsum('bhld,bhLd->bhlL', q, k) / jnp.sqrt(d_h)
        attn = nn.softmax(attn, -1)
        y = jnp.einsum('bhll,bhLd->bhld', attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(b, l, d)
        return nn.Dense(hidden_dim, use_bias=True, name='out')(y)

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