from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import flax.linen as nn
from flax.linen import dot_product_attention
from einops import rearrange
import math

from bageljax.model.streaming_attention import streaming_attention
from bageljax.utils.jax_utils import add_batch_sharding_constraint, is_sharding_active, get_current_mesh
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

class PatchEmbed(nn.Module):
    """14 × 14 patchifier → (B, L, C)."""
    embed_dim: int = 1152
    patch_size: int = 14
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x): # we assume input images are already normalized to [-1, 1]
        x = add_batch_sharding_constraint(x, where="input to patch embed")
        x = x.astype(self.param_dtype) # cast to bfloat16 if not already
        x = rearrange(x, 'b (hn hp) (wn wp) c -> b hn hp wn wp c', hp=self.patch_size, wp=self.patch_size)
        x = rearrange(x, 'b hn hp wn wp c -> b hn wn (hp wp c)')
        _, h, w, _ = x.shape
        x = rearrange(x, 'b h w d -> b (h w) d')  # (B, L, C)
        x = nn.Dense(self.embed_dim, use_bias=True, name='proj', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        x = add_batch_sharding_constraint(x, where="output of patch embed")
        return x, (h, w)
    
class MHA(nn.Module):
    """Multi-head self-attention (flash/SDPA via `dot_product_attention`)."""
    heads: int = 16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,                 # (B, L, D)
                 ):
        x = add_batch_sharding_constraint(x, where="input to ViT MHA")
        B, L, D = x.shape
        H       = self.heads
        d_h     = D // H

        # ——— ensure activations arrive in bf16 ———
        x = x.astype(self.param_dtype)

        dense = lambda name: nn.Dense(D, use_bias=True, name=name, dtype=self.param_dtype, param_dtype=self.param_dtype)

        # -------- QKV projections ----------------------------------------------------
        q = jnp.transpose(dense('q')(x).reshape(B, L, H, d_h), (0, 2, 1, 3))
        k = jnp.transpose(dense('k')(x).reshape(B, L, H, d_h), (0, 2, 1, 3))
        v = jnp.transpose(dense('v')(x).reshape(B, L, H, d_h), (0, 2, 1, 3))

        q = add_batch_sharding_constraint(q, where="q in ViT MHA")
        k = add_batch_sharding_constraint(k, where="k in ViT MHA")
        v = add_batch_sharding_constraint(v, where="v in ViT MHA")

        # Attention. We set force_fp32_for_softmax=True following standard practice
        #y = dot_product_attention(
        #        query=q,
        #        key=k,
        #        value=v,
        #        mask=None,
        #        dropout_rate=0.0,
        #        deterministic=True,
        #        force_fp32_for_softmax=True,
        #)

        #y = streaming_attention(
        #    q, k, v,
        #    bias=None,
        #    block_size=128,
        #    out_dtype=jnp.bfloat16,
        #    accum_dtype=jnp.float32,
        #)

        sharding_active = is_sharding_active()

        if sharding_active:
            mesh = get_current_mesh()

            def _fa(q_, k_, v_):
                return flash_attention(q_, k_, v_, sm_scale=1.0/math.sqrt(d_h),)

            in_specs  = (P('devices', None, None, None),
                        P('devices', None, None, None),
                        P('devices', None, None, None))
            out_specs = P('devices', None, None, None)

            y = shard_map(_fa, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)(q, k, v)

        else:
            y = flash_attention(
                q, k, v,
                sm_scale=1.0/math.sqrt(d_h),
            )

        # -------- merge heads & output projection -----------------------------------
        y = jnp.transpose(y, (0, 2, 1, 3))
        y = y.reshape(B, L, D)
        y = add_batch_sharding_constraint(y, where="attention output in ViT")
        return nn.Dense(D, use_bias=True, name='out', dtype=self.param_dtype, param_dtype=self.param_dtype)(y)

class MLP(nn.Module):
    projection_dim: int = 4304
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = add_batch_sharding_constraint(x, where="input to ViT MLP")
        x = x.astype(self.param_dtype)
        hidden_dim = x.shape[-1]
        x = nn.Dense(self.projection_dim, use_bias=True, name='fc1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(hidden_dim, use_bias=True, name='fc2', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        x = add_batch_sharding_constraint(x, where="output of ViT MLP")
        return x

class EncoderBlock(nn.Module):
    heads: int = 16
    projection_dim: int = 4304
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = add_batch_sharding_constraint(x, where="input to ViT encoder block")
        x = x.astype(self.param_dtype)

        # attention
        h = nn.LayerNorm(name='ln1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = MHA(self.heads, name='mha')(h)
        x = x + h
        x = add_batch_sharding_constraint(x, where="middle of ViT encoder block")
        # MLP
        h = nn.LayerNorm(name='ln2', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = MLP(self.projection_dim, name='mlp')(h)
        return add_batch_sharding_constraint(x + h, where="output of ViT encoder block")

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
        x = add_batch_sharding_constraint(x, where="in NaViT")

        # transformer encoder
        for i in range(self.depth):
            x = EncoderBlock(self.heads, self.projection_dim, name=f'block_{i}')(x)
        x = nn.LayerNorm(name='post_vit_ln', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        x = add_batch_sharding_constraint(x, where="output of NaViT")
        return x, (hp, wp)

class ViTConnector(nn.Module):
    in_dim: int = 1152
    out_dim: int = 3584
    max_grid: int = 70
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, grid_hw):
        x = add_batch_sharding_constraint(x, where="input to ViTConnector")
        hp, wp = grid_hw
        # 2-layer GELU MLP
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc1', dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.out_dim, use_bias=True, name='connector_fc2', dtype=self.param_dtype, param_dtype=self.param_dtype)(h)

        pos_xmodal = self.param('post_vit_pos_embed',
                                nn.initializers.normal(0.02, dtype=self.param_dtype),
                                (self.max_grid ** 2, self.out_dim))
        idx = jnp.arange(hp)[:, None] * self.max_grid + jnp.arange(wp)[None, :]
        return h + jax.lax.stop_gradient(jnp.take(pos_xmodal, idx.reshape(-1), axis=0)) # these embeddings were never learnt, but rather initialized via 2D sin/cos

class VisionEncoder(nn.Module):
    """NHWC pixels  → patch tokens in LLM space (3584-d)."""
    depth: int = 26

    @nn.compact
    def __call__(self, img):
        vit_tokens, hw = NaViT(self.depth, name='vit')(img)
        llm_tokens = ViTConnector(name='connector')(vit_tokens, hw)
        llm_tokens = add_batch_sharding_constraint(llm_tokens, where="output of vision encoder")
        return llm_tokens  # (B, L, 3584)
