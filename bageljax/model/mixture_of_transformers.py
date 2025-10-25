from __future__ import annotations
from typing import Optional, Tuple
import functools as ft
import jax
import jax.numpy as jnp
import jax.lax   as lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import flax.linen as nn
from flax.linen import dot_product_attention
from flax.core import broadcast
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

from bageljax.utils.jax_utils import add_batch_sharding_constraint
from bageljax.utils.jax_utils import get_current_mesh, is_sharding_active

# A more memory saving remat policy than Jax's default if you want to use
from jax import checkpoint_policies as cpp
REMAT_POLICY = cpp.save_anything_except_these_names(
    "dot_general",            # most matmuls (einsum lowers here, too)
    "dot",                    # legacy small dot
)

class RMSNorm(nn.Module):
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Record the dtype of the input
        input_dtype = x.dtype

        x = add_batch_sharding_constraint(x, where="input to RMSNorm")

        w = self.param(
            "weight",
            ft.partial(nn.initializers.ones, dtype=self.param_dtype),
            (x.shape[-1],)
        )

        # Compute 1/√(mean(x²) + ε) in fp32
        inv_rms = jax.lax.rsqrt(
            jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            + self.eps
        ).astype(input_dtype)

        return add_batch_sharding_constraint(x * (w * inv_rms), where="output of rms norm")


def rotary_cache(L: int, dim: int = 128, base: float = 1_000_000.0):
    half = dim // 2
    exponents = jnp.arange(half, dtype=jnp.float32) / half # rope cos and sin should be originated in float32
    inv_freq = (base ** (-exponents)).astype(jnp.float32)
    inv_freq_expanded = inv_freq[:, None]
    max_seq_len_position_ids = jnp.arange(L, dtype=jnp.float32)[None, :]
    freqs = jnp.matmul(inv_freq_expanded, max_seq_len_position_ids).T
    emb = jnp.concatenate([freqs, freqs], axis=-1) # (L, dim)
    return jnp.cos(emb), jnp.sin(emb)

def apply_rope(
    x:  jnp.ndarray,        # (B, L, H, D)   – bf16
    pos: jnp.ndarray,       # (B, L)         – int32 indices
    cos: jnp.ndarray,       # (max_pos, D)   fp32
    sin: jnp.ndarray,       # (max_pos, D)   fp32
) -> jnp.ndarray:
    x = add_batch_sharding_constraint(x, where="input to apply rope")

    # Select from cos, sin according to pos
    cos, sin = jnp.take(cos, pos, axis=0), jnp.take(sin, pos, axis=0) # each will have resultant shape (B, L, D)

    # Add head dim to cos/sin
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]

    # Downcast cos/sin to bfloat16
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return jnp.concatenate([-x2, x1], axis=-1)

    y = (x * cos) + (rotate_half(x) * sin)
    y = add_batch_sharding_constraint(y, where="output of apply rope")
    return y


class SwiGLU(nn.Module):
    hidden:  int = 3584
    inner:   int = 18_944
    expert:  str = "txt"     # "txt" | "gen"
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.gate = nn.Dense(self.inner, use_bias=False,  name=f"{self.expert}/gate_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.up   = nn.Dense(self.inner, use_bias=False,  name=f"{self.expert}/up_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.down = nn.Dense(self.hidden, use_bias=False, name=f"{self.expert}/down_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)

    def __call__(self, x):                 # (B,L,hidden)
        x = add_batch_sharding_constraint(x, where="input of swiglu")
        x = x.astype(self.param_dtype) # ensure bf16
        x = self.down(nn.swish(self.gate(x)) * self.up(x))
        return add_batch_sharding_constraint(x, where="output of swiglu")


class GQA(nn.Module):
    heads:      int = 28
    kv_heads:   int = 4
    hidden:     int = 3584
    rope_dim:   int = 128
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        # txt expert
        self.q_txt = nn.Dense(self.hidden, use_bias=True,  name="txt/q_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.k_txt = nn.Dense(512,        use_bias=True,  name="txt/k_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.v_txt = nn.Dense(512,        use_bias=True,  name="txt/v_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.o_txt = nn.Dense(self.hidden, use_bias=False, name="txt/o_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)

        # QK norm parameters
        self.q_norm_txt = RMSNorm(name="txt/q_norm")
        self.k_norm_txt = RMSNorm(name="txt/k_norm")

    def __call__(self,
                 x: jnp.ndarray,              # (B,L,3584)
                 *,
                 rope_pos_ids: jnp.ndarray,   # (B,L) int32
                 attn_bias: jnp.ndarray,      # (B,1,L,L)  large-neg where masked
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                ):
        x = add_batch_sharding_constraint(x, where="x, call to attenion")
        rope_pos_ids = add_batch_sharding_constraint(rope_pos_ids, where="rope_pos_ids, call to attention")
        attn_bias = add_batch_sharding_constraint(attn_bias, where="attn_bias, call to attention")

        # ensure bf16
        x = x.astype(self.param_dtype)
        attn_bias = attn_bias.astype(self.param_dtype)

        B, L, _ = x.shape
        H, H_kv = self.heads, self.kv_heads
        d_q  = self.hidden // H            # 128 == rope_dim
        d_kv = 512 // H_kv                 # 128 == rope_dim

        # ----------- projections --------------
        q, k, v = self.q_txt(x), self.k_txt(x), self.v_txt(x)

        # reshape to heads first
        def split_heads(t, n_head, d_head):
            return t.reshape(B, L, n_head, d_head)

        q = split_heads(q, H,    d_q)
        k = split_heads(k, H_kv, d_kv)
        v = split_heads(v, H_kv, d_kv)

        # -------------- QK Norm -------------------
        q = self.q_norm_txt(q)
        k = self.k_norm_txt(k)

        q = add_batch_sharding_constraint(q, where="q1")
        k = add_batch_sharding_constraint(k, where="k1")
        v = add_batch_sharding_constraint(v, where="v1")
        
        # ----------- RoPE --------------
        q = apply_rope(q, rope_pos_ids, cos, sin)
        k = apply_rope(k, rope_pos_ids, cos, sin)

        # ----------- broadcast KV heads to match Q heads (GQA) ------------
        rep = H // H_kv
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)     # shapes (B,L,H,128)

        # Transpose to prepare for call to flash attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Tile attn bias on head dimension
        attn_bias = jnp.tile(attn_bias, (1, H, 1, 1))
        attn_bias = add_batch_sharding_constraint(attn_bias, where="tiling attn bias")

        q = add_batch_sharding_constraint(q, where="q2")
        k = add_batch_sharding_constraint(k, where="k2")
        v = add_batch_sharding_constraint(v, where="v2")

        # ---------- flash attention -------------------
        sharding_active = is_sharding_active()

        if sharding_active:
            mesh = get_current_mesh()

            def _fa(q_, k_, v_, b_):
                return flash_attention(q_, k_, v_, b_, sm_scale=0.08838834764,)

            in_specs  = (P('devices', None, None, None),
                        P('devices', None, None, None),
                        P('devices', None, None, None),
                        P('devices', None, None, None))
            out_specs = P('devices', None, None, None)

            out = shard_map(_fa, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)(q, k, v, attn_bias)

        else:
            out = flash_attention(
                q, k, v,
                attn_bias,
                sm_scale=0.08838834764,
            )

        # ---------- merge heads & out-proj -----------------
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(B, L, self.hidden)   # (B,L,3584)

        # out might be in float32, but the projection below will convert to bf16 since we specified dtype=bfloat16 in its definition

        out = self.o_txt(out) # (B,L,3584)
        out = add_batch_sharding_constraint(out, where="output of GQA")

        return out                   


class MoTBlock(nn.Module):
    hidden: int = 3584

    def setup(self):
        # RMSNorm banks
        self.in_rms_txt  = RMSNorm(name="txt/input_rms")
        self.post_rms_txt = RMSNorm(name="txt/post_attn_rms")

        # Attention & MLP
        self.attn = GQA(name="attn")
        self.mlp_txt = SwiGLU(expert="txt", name="txt/mlp")

    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 rope_pos_ids: jnp.ndarray,
                 attn_bias: jnp.ndarray,
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                ):
        x = add_batch_sharding_constraint(x, where="x, input to MoTBlock")
        rope_pos_ids = add_batch_sharding_constraint(rope_pos_ids, where="rope_pos_ids, input to MoTBlock")
        attn_bias = add_batch_sharding_constraint(attn_bias, where="attn_bias, input to MoTBlock")

        # ensure bf16. Redundant, but let's do it anyway
        x = x.astype(jnp.bfloat16)
        attn_bias = attn_bias.astype(jnp.bfloat16)

        # RMS
        h = self.in_rms_txt(x)

        # shared attention
        h = self.attn(h,
                      rope_pos_ids=rope_pos_ids,
                      attn_bias=attn_bias,
                      cos=cos, sin=sin,
                      )
        h = add_batch_sharding_constraint(h, where="output of attention")
        x = x + h
        x = add_batch_sharding_constraint(x, where="after skip connection")

        # post-RMS
        h = self.post_rms_txt(x)

        # expert-specific MLP
        h = self.mlp_txt(h)
        h = add_batch_sharding_constraint(h, where="after mlp")

        return x + h

class MoTStep(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self,
                 carry: jnp.ndarray,          # carry = x
                 rope_pos_ids: jnp.ndarray,
                 attn_bias: jnp.ndarray,
                 cos: jnp.ndarray,
                 sin: jnp.ndarray):
        x = MoTBlock(hidden=self.hidden)(
            carry,
            rope_pos_ids=rope_pos_ids,
            attn_bias=attn_bias,
            cos=cos, sin=sin
        )
        # force no per-step outputs
        return x, None

class MixtureOfTransformers(nn.Module):
    depth:      int = 28
    hidden:     int = 3584
    rope_dim:   int = 128

    def setup(self):
        self.out_rms_txt = RMSNorm(name="txt/final_rms")

    @nn.compact
    def __call__(self, x, rope_pos_ids, attn_bias):
        x = add_batch_sharding_constraint(x, where="x, input to MoT")
        rope_pos_ids = add_batch_sharding_constraint(rope_pos_ids, where="rope_pos_ids, input to MoT")
        attn_bias = add_batch_sharding_constraint(attn_bias, where="attn_bias, input to MoT")

        x = x.astype(jnp.bfloat16)
        attn_bias = attn_bias.astype(jnp.bfloat16)

        B, L, _ = x.shape

        # RoPE tables (keep in f32; scan inputs are broadcast, so cheap)
        cos, sin = rotary_cache(L, self.rope_dim)

        ScannedMoT = nn.scan(
            # You can checkpoint to reduce compile-time RAM:
            nn.remat(MoTStep),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(broadcast, broadcast, broadcast, broadcast),   # four separate broadcast xs
            out_axes=None,                      # ys=None from the step
            length=self.depth,                  # <-- set length here
            unroll=1,
        )

        layers = ScannedMoT(name="layers", hidden=self.hidden)

        # Call with carry first, then the four xs (no length kwarg here)
        x, _ = layers(x, rope_pos_ids, attn_bias, cos, sin)

        # ----- final norm -----
        x = self.out_rms_txt(x)
        x = add_batch_sharding_constraint(x, where="output of MoT")
        return x
