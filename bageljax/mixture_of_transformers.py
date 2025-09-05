from __future__ import annotations
from typing import Optional, Tuple
import functools as ft

import jax
import jax.numpy as jnp
import jax.lax   as lax
import flax.linen as nn
from flax.linen import dot_product_attention


class RMSNorm(nn.Module):
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Record the dtype of the input
        input_dtype = x.dtype

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

        return x * (w * inv_rms)


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
        x = x.astype(self.param_dtype) # ensure bf16
        return self.down(nn.swish(self.gate(x)) * self.up(x))


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

        # gen (VAE) expert
        self.q_gen = nn.Dense(self.hidden, use_bias=True,  name="gen/q_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.k_gen = nn.Dense(512,        use_bias=True,  name="gen/k_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.v_gen = nn.Dense(512,        use_bias=True,  name="gen/v_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)
        self.o_gen = nn.Dense(self.hidden, use_bias=False, name="gen/o_proj", dtype=self.param_dtype, param_dtype=self.param_dtype)

        # QK norm parameters
        self.q_norm_txt = RMSNorm(name="txt/q_norm")
        self.k_norm_txt = RMSNorm(name="txt/k_norm")
        self.q_norm_gen = RMSNorm(name="gen/q_norm")
        self.k_norm_gen = RMSNorm(name="gen/k_norm")

    def __call__(self,
                 x: jnp.ndarray,              # (B,L,3584)
                 *,
                 token_types: jnp.ndarray,    # (B,L) 0=pad, 1=text/img, 2=vae
                 rope_pos_ids: jnp.ndarray,   # (B,L) int32
                 attn_bias: jnp.ndarray,      # (B,1,L,L)  large-neg where masked
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                ):
        # ensure bf16
        x = x.astype(self.param_dtype)
        attn_bias = attn_bias.astype(self.param_dtype)

        B, L, _ = x.shape
        H, H_kv = self.heads, self.kv_heads
        d_q  = self.hidden // H            # 128 == rope_dim
        d_kv = 512 // H_kv                 # 128 == rope_dim

        mask_gen = (token_types == 2)      # (B,L) bool

        # ----------- projections (compute both, then select) --------------
        q_txt, k_txt, v_txt = self.q_txt(x), self.k_txt(x), self.v_txt(x)
        q_gen, k_gen, v_gen = self.q_gen(x), self.k_gen(x), self.v_gen(x)

        sel   = mask_gen[..., None]        # broadcast dim
        q = jnp.where(sel, q_gen, q_txt)
        k = jnp.where(sel, k_gen, k_txt)
        v = jnp.where(sel, v_gen, v_txt)

        # reshape to heads first
        def split_heads(t, n_head, d_head):
            return t.reshape(B, L, n_head, d_head)

        q = split_heads(q, H,    d_q)
        k = split_heads(k, H_kv, d_kv)
        v = split_heads(v, H_kv, d_kv)

        # -------------- QK Norm ----------------------
        sel_h = mask_gen[:, :, None, None]                 # (B, L, 1, 1)

        q = jnp.where(
            sel_h,                         
            self.q_norm_gen(q),
            self.q_norm_txt(q),
        )

        k = jnp.where(
            sel_h,
            self.k_norm_gen(k),
            self.k_norm_txt(k),
        )
        
        # ----------- RoPE --------------
        q = apply_rope(q, rope_pos_ids, cos, sin)
        k = apply_rope(k, rope_pos_ids, cos, sin)

        # ----------- broadcast KV heads to match Q heads (GQA) ------------
        rep = H // H_kv
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)     # shapes (B,L,H,128)

        # ---------- dot-product attention -------------------
        out = dot_product_attention(
            q, k, v,
            bias=attn_bias,
            dropout_rate=0.0,
            deterministic=True,
            force_fp32_for_softmax=True,
        )

        # ---------- merge heads & expert-specific out-proj -----------------
        out = out.reshape(B, L, self.hidden)   # (B,L,3584)

        # out might be in float32, but the projections below will convert to bf16 since we specified dtype=bfloat16 in their definitions

        out_txt = self.o_txt(out)
        out_gen = self.o_gen(out)
        return jnp.where(sel, out_gen, out_txt)                   # (B,L,3584)


class MoTBlock(nn.Module):
    hidden: int = 3584

    def setup(self):
        # RMSNorm banks
        self.in_rms_txt  = RMSNorm(name="txt/input_rms")
        self.in_rms_gen  = RMSNorm(name="gen/input_rms")
        self.post_rms_txt = RMSNorm(name="txt/post_attn_rms")
        self.post_rms_gen = RMSNorm(name="gen/post_attn_rms")

        # Attention & MLP
        self.attn = GQA(name="attn")
        self.mlp_txt = SwiGLU(expert="txt", name="txt/mlp")
        self.mlp_gen = SwiGLU(expert="gen", name="gen/mlp")

    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 token_types: jnp.ndarray,
                 rope_pos_ids: jnp.ndarray,
                 attn_bias: jnp.ndarray,
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                ):
        # ensure bf16. Redundant, but let's do it anyway
        x = x.astype(jnp.bfloat16)
        attn_bias = attn_bias.astype(jnp.bfloat16)

        mask_gen = (token_types == 2)[..., None]          # (B,L,1)

        # expert-specific input RMS
        h = jnp.where(mask_gen, self.in_rms_gen(x), self.in_rms_txt(x))

        # shared attention
        h = self.attn(h,
                      token_types=token_types,
                      rope_pos_ids=rope_pos_ids,
                      attn_bias=attn_bias,
                      cos=cos, sin=sin,
                      )
        x = x + h

        # expert-specific post-RMS
        h = jnp.where(mask_gen, self.post_rms_gen(x), self.post_rms_txt(x))

        # expert-specific MLP
        h = jnp.where(mask_gen, self.mlp_gen(h), self.mlp_txt(h))

        return x + h


class MixtureOfTransformers(nn.Module):
    depth:      int = 28
    hidden:     int = 3584
    rope_dim:   int = 128

    def setup(self):
        self.blocks = [MoTBlock(name=f"layer_{i}") for i in range(self.depth)]
        self.out_rms_txt = RMSNorm(name="txt/final_rms")
        self.out_rms_gen = RMSNorm(name="gen/final_rms")

    def __call__(self,
                 x: jnp.ndarray,                  # (B,L,3584)
                 token_types: jnp.ndarray,         # (B,L) 0=pad, 1=text/img, 2=vae
                 rope_pos_ids: jnp.ndarray,
                 attn_bias: jnp.ndarray,
                ):     
        # ensure bf16 (redundant, but let's be sure)
        x = x.astype(jnp.bfloat16) 
        attn_bias = attn_bias.astype(jnp.bfloat16)

        B, L, _ = x.shape

        # ---- RoPE tables -------------------------------------------------
        cos, sin = rotary_cache(L, self.rope_dim)  # these will be in float32

        # ---- transformer stack ------------------------------------------
        for blk in self.blocks:
            x = blk(x,
                    token_types=token_types,
                    rope_pos_ids=rope_pos_ids,
                    attn_bias=attn_bias,
                    cos=cos, sin=sin)

        # ---- final norm per expert --------------------------------------
        mask_gen = (token_types == 2)[..., None]
        x = jnp.where(mask_gen, self.out_rms_gen(x), self.out_rms_txt(x))
        return x
