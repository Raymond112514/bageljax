# mixture_of_transformers.py
# ================================================================
#  Mixture-of-Two Qwen-style Transformer stacks (26 layers, d=3584)
#  * One common attention operation → full cross-modal interaction
#  * Two complete *expert* parameter banks (txt  vs  gen/vae)
#  * GQA: 28 Q-heads (224-d)  /  4 KV-heads (128-d)
#  * RoPE positional encoding
# ================================================================
from __future__ import annotations
from typing import Optional, Tuple

import jax.numpy as jnp
import jax.lax   as lax
import flax.linen as nn
from flax.linen import dot_product_attention


# -----------------------------------------------------------------
#  Norm – RMSNorm (weight only, no bias)
# -----------------------------------------------------------------
class RMSNorm(nn.Module):
    eps: float = 1e-6
    def __call__(self, x):
        w = self.param("weight", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * (w / rms)


# -----------------------------------------------------------------
#  Rotary helpers
# -----------------------------------------------------------------
def rotary_cache(L: int, dim: int = 128, base: int = 10_000):
    freqs  = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
    t      = jnp.arange(L)
    angles = t[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)             # (L,dim//2)

def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    # x shape (..., L, dim)  – dim even and == cos*2
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotx1  = x1 * cos - x2 * sin
    rotx2  = x1 * sin + x2 * cos
    x_rot  = jnp.stack([rotx1, rotx2], axis=-1).reshape(x.shape)
    return x_rot


# -----------------------------------------------------------------
#  SwiGLU MLP — duplicated params per expert
# -----------------------------------------------------------------
class SwiGLU(nn.Module):
    hidden:  int = 3584
    inner:   int = 18_944
    expert:  str = "txt"     # "txt" | "gen"

    def setup(self):
        self.gate = nn.Dense(self.inner, use_bias=False,  name=f"{self.expert}/gate_proj")
        self.up   = nn.Dense(self.inner, use_bias=False,  name=f"{self.expert}/up_proj")
        self.down = nn.Dense(self.hidden, use_bias=False, name=f"{self.expert}/down_proj")

    def __call__(self, x):                 # (B,L,hidden)
        return self.down(nn.swish(self.gate(x)) * self.up(x))


# -----------------------------------------------------------------
#  GQA attention with per-token *expert* selection
# -----------------------------------------------------------------
class GQAAttn(nn.Module):
    heads:      int = 28
    kv_heads:   int = 4
    hidden:     int = 3584
    rope_dim:   int = 128

    # ---- parameter banks -------------------------------------------------
    def setup(self):
        # txt expert
        self.q_txt = nn.Dense(self.hidden, use_bias=True,  name="txt/q_proj")
        self.k_txt = nn.Dense(512,        use_bias=True,  name="txt/k_proj")
        self.v_txt = nn.Dense(512,        use_bias=True,  name="txt/v_proj")
        self.o_txt = nn.Dense(self.hidden, use_bias=False, name="txt/o_proj")

        # gen (VAE) expert
        self.q_gen = nn.Dense(self.hidden, use_bias=True,  name="gen/q_proj")
        self.k_gen = nn.Dense(512,        use_bias=True,  name="gen/k_proj")
        self.v_gen = nn.Dense(512,        use_bias=True,  name="gen/v_proj")
        self.o_gen = nn.Dense(self.hidden, use_bias=False, name="gen/o_proj")

        # learned rescalers for Q,K  (shape 128)  – 1 per expert
        init = nn.initializers.ones
        self.q_norm_txt = self.param("txt/q_norm", init, (self.rope_dim,))
        self.k_norm_txt = self.param("txt/k_norm", init, (self.rope_dim,))
        self.q_norm_gen = self.param("gen/q_norm", init, (self.rope_dim,))
        self.k_norm_gen = self.param("gen/k_norm", init, (self.rope_dim,))

    # ---- forward ---------------------------------------------------------
    def __call__(self,
                 x: jnp.ndarray,              # (B,L,3584)
                 *,
                 token_types: jnp.ndarray,    # (B,L) 0=pad, 1=text/img, 2=vae
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                 attn_bias: jnp.ndarray,      # (B,1,L,L)  large-neg where masked
                 deterministic: bool):

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

        # ---------- apply per-expert head-dim rescale ----------------------
        # reshape to heads first
        def split_heads(t, n_head, d_head):
            return t.reshape(B, L, n_head, d_head).transpose(0,2,1,3) # we transpose here because rope expects (..., L, dim)

        q = split_heads(q, H,    d_q)
        k = split_heads(k, H_kv, d_kv)
        v = split_heads(v, H_kv, d_kv)

        # ----------- RoPE (only first 128 dims of each head) --------------
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK norm, goes after RoPE
        sel_h = mask_gen[:, None, :, None]                 # (B, 1, L, 1)

        q_scale = jnp.where(
            sel_h,                                         # broadcast on head axis
            self.q_norm_gen[None, None, None, :],          # (1,1,1,d_q)
            self.q_norm_txt[None, None, None, :],
        )                                                  # (B,1,L,d_q) → broadcast

        k_scale = jnp.where(
            sel_h,
            self.k_norm_gen[None, None, None, :],
            self.k_norm_txt[None, None, None, :],
        )

        q = q * q_scale                                    # (B, H,    L, d_q)
        k = k * k_scale                                    # (B, H_kv, L, d_kv)

        # ----------- broadcast KV heads to match Q heads (GQA) ------------
        rep = H // H_kv
        k = jnp.repeat(k, rep, axis=1)
        v = jnp.repeat(v, rep, axis=1)     # shapes (B,H,L,128)

        # ---------- dot-product attention (FLASH fused) -------------------
        q = q.transpose(0,2,1,3) # (B,L,H,128)
        k = k.transpose(0,2,1,3)
        v = v.transpose(0,2,1,3)

        out = dot_product_attention(
                q, k, v,
                bias=attn_bias,
                dropout_rate=0.0,
                deterministic=deterministic,
                dtype=q.dtype,
                precision='highest') # highest allows for more stable attention

        # ---------- merge heads & expert-specific out-proj -----------------
        out = out.reshape(B, L, self.hidden)   # (B,L,3584)

        out_txt = self.o_txt(out)
        out_gen = self.o_gen(out)
        return jnp.where(sel, out_gen, out_txt)                   # (B,L,3584)


# -----------------------------------------------------------------
#  One *shared-attention* block with expert-specific params
# -----------------------------------------------------------------
class MoTBlock(nn.Module):
    hidden: int = 3584

    def setup(self):
        # RMSNorm banks
        self.in_rms_txt  = RMSNorm(name="txt/input_rms")
        self.in_rms_gen  = RMSNorm(name="gen/input_rms")
        self.post_rms_txt = RMSNorm(name="txt/post_attn_rms")
        self.post_rms_gen = RMSNorm(name="gen/post_attn_rms")

        # Attention & MLP
        self.attn = GQAAttn(name="attn")
        self.mlp_txt = SwiGLU(expert="txt", name="txt/mlp")
        self.mlp_gen = SwiGLU(expert="gen", name="gen/mlp")

    # ------------------------------------------------------------------
    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 token_types: jnp.ndarray,
                 cos: jnp.ndarray,
                 sin: jnp.ndarray,
                 attn_bias: jnp.ndarray,
                 deterministic: bool):

        mask_gen = (token_types == 2)[..., None]          # (B,L,1)

        # expert-specific input RMS
        h = jnp.where(mask_gen, self.in_rms_gen(x), self.in_rms_txt(x))

        # shared attention
        h = self.attn(h,
                      token_types=token_types,
                      cos=cos, sin=sin,
                      attn_bias=attn_bias,
                      deterministic=deterministic)
        x = x + h

        # expert-specific post-RMS
        h = jnp.where(mask_gen, self.post_rms_gen(x), self.post_rms_txt(x))

        # expert-specific MLP
        h_txt = self.mlp_txt(h)
        h_gen = self.mlp_gen(h)
        h = jnp.where(mask_gen, h_gen, h_txt)

        return x + h


# -----------------------------------------------------------------
#  Full stack (26 layers) + final RMSNorm
# -----------------------------------------------------------------
class MixtureOfTransformers(nn.Module):
    depth:      int = 26
    hidden:     int = 3584
    rope_dim:   int = 128

    def setup(self):
        self.blocks = [MoTBlock(name=f"layer_{i}") for i in range(self.depth)]
        self.out_rms_txt = RMSNorm(name="txt/final_rms")
        self.out_rms_gen = RMSNorm(name="gen/final_rms")

    # ------------------------------------------------------------------
    def __call__(self,
                 x: jnp.ndarray,                  # (B,L,3584)
                 *,
                 token_types: jnp.ndarray,        # (B,L) 0=pad, 1=text/img, 2=vae
                 deterministic: bool = True):

        B, L, _ = x.shape

        # ---- construct attention bias -----------------------------------
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        is_vae = (token_types == 2)
        # allow VAE queries to see all VAE keys
        allowed = jnp.where(is_vae[:, :, None],
                            causal[None],              # broadcast B
                            causal[None])
        allowed = jnp.where(is_vae[:, :, None] & is_vae[:, None, :],
                    True,
                    allowed)
        # padding (token_type==0) sees nothing, and is seen by nothing
        allowed = jnp.where((token_types[:, :, None] == 0) | (token_types[:, None, :] == 0), False, allowed)

        attn_bias = jnp.where(allowed, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)

        # ---- RoPE tables -------------------------------------------------
        cos, sin = rotary_cache(L, self.rope_dim)

        # ---- transformer stack ------------------------------------------
        for blk in self.blocks:
            x = blk(x,
                    token_types=token_types,
                    cos=cos, sin=sin,
                    attn_bias=attn_bias,
                    deterministic=deterministic)

        # ---- final norm per expert --------------------------------------
        mask_gen = (token_types == 2)[..., None]
        x = jnp.where(mask_gen, self.out_rms_gen(x), self.out_rms_txt(x))
        return x
