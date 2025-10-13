import jax
import jax.numpy as jnp
import jax.lax as lax

def streaming_attention(
    q: jnp.ndarray,            # (B, Lq, H, Dh)
    k: jnp.ndarray,            # (B, Lk, H, Dh)
    v: jnp.ndarray,            # (B, Lk, H, Dh)
    bias: jnp.ndarray | None,  # None or (B, 1, Lq, Lk) ; for mask-only use entries <= neg_mask_threshold as masked
    *,
    block_size: int = 256,
    out_dtype: jnp.dtype = jnp.bfloat16,
    accum_dtype: jnp.dtype = jnp.float32,
    bias_is_mask: bool = True,
    neg_mask_threshold: float = -1e20,
) -> jnp.ndarray:
    assert q.ndim == k.ndim == v.ndim == 4
    B, Lq, H, Dh = q.shape
    assert k.shape == (B, k.shape[1], H, Dh) and v.shape == (B, k.shape[1], H, Dh)
    Lk = k.shape[1]

    # Work in (B,H,...) so bias (B,1,Lq,chunk) broadcasts implicitly on H
    q_ = jnp.transpose(q, (0, 2, 1, 3)).astype(accum_dtype)  # (B,H,Lq,Dh)
    k_ = jnp.transpose(k, (0, 2, 1, 3)).astype(accum_dtype)  # (B,H,Lk,Dh)
    v_ = jnp.transpose(v, (0, 2, 1, 3)).astype(accum_dtype)  # (B,H,Lk,Dh)

    # Pad K/V (and bias if present) so slices have static size = block_size
    pad_cols = (-Lk) % block_size
    if pad_cols:
        k_ = jnp.pad(k_, ((0,0),(0,0),(0,pad_cols),(0,0)), mode="constant", constant_values=0)
        v_ = jnp.pad(v_, ((0,0),(0,0),(0,pad_cols),(0,0)), mode="constant", constant_values=0)
    Lk_pad = Lk + pad_cols
    nblocks = Lk_pad // block_size

    have_bias = bias is not None
    if have_bias:
        assert bias.shape == (B, 1, Lq, Lk)
        pad_val = (neg_mask_threshold if bias_is_mask else -1e30)
        if pad_cols:
            bias = jnp.pad(bias.astype(accum_dtype),
                           ((0,0),(0,0),(0,0),(0,pad_cols)),
                           mode="constant",
                           constant_values=pad_val)
        else:
            bias = bias.astype(accum_dtype)

    scale = jnp.asarray(1.0 / jnp.sqrt(Dh), dtype=accum_dtype)

    # Running stats per (B,H,Lq)
    m = jnp.full((B, H, Lq), -jnp.inf, dtype=accum_dtype)   # running max
    l = jnp.zeros((B, H, Lq), dtype=accum_dtype)            # running sumexp
    o = jnp.zeros((B, H, Lq, Dh), dtype=accum_dtype)        # running numerator

    # Small constant used inside the loop only (safe)
    col_idx = jnp.arange(block_size, dtype=jnp.int32)       # (block,)

    def body(carry, _unused):
        m, l, o, start = carry  # carry start, don’t pass bi via xs
        # Slice K_i, V_i : (B,H,block,Dh)
        k_i = lax.dynamic_slice(k_, (0, 0, start, 0), (B, H, block_size, Dh))
        v_i = lax.dynamic_slice(v_, (0, 0, start, 0), (B, H, block_size, Dh))

        # Scores: (B,H,Lq,block)
        s_i = jnp.matmul(q_, jnp.swapaxes(k_i, -1, -2)) * scale

        # Valid columns for this chunk (tail-safe), computed from carried 'start'
        valid_cols = (start + col_idx) < Lk  # (block,)

        if have_bias:
            # (B,1,Lq,block) slice; implicit broadcast over H inside ops
            b_i = lax.dynamic_slice(bias, (0, 0, 0, start), (B, 1, Lq, block_size))
            if bias_is_mask:
                # Only use bias as a mask; DO NOT add to scores (avoids big fp32 broadcasts).
                allowed = (b_i > neg_mask_threshold) & valid_cols[None, None, None, :]
            else:
                # General additive bias path
                s_i = s_i + b_i
                allowed = valid_cols[None, None, None, :]
        else:
            allowed = valid_cols[None, None, None, :]

        # Mask before softmax
        s_i = jnp.where(allowed, s_i, -jnp.inf)

        # Stable running softmax per-row over 'block'
        m_i   = jnp.max(s_i, axis=-1)                  # (B,H,Lq)
        all_invalid_chunk = jnp.isneginf(m_i)

        m_cand = jnp.maximum(m, m_i)
        exp_m  = jnp.exp(m - m_cand)                   # (B,H,Lq)
        exp_i  = jnp.exp(s_i - m_cand[..., None])      # (B,H,Lq,block)
        l_cand = exp_m * l + jnp.sum(exp_i, axis=-1)   # (B,H,Lq)
        o_cand = exp_m[..., None] * o + jnp.matmul(exp_i, v_i)

        m = jnp.where(all_invalid_chunk, m, m_cand)
        l = jnp.where(all_invalid_chunk, l, l_cand)
        o = jnp.where(all_invalid_chunk[..., None], o, o_cand)

        return (m, l, o, start + block_size), None

    # IMPORTANT: scan with xs=None and carry 'start', and force unroll=1
    (m, l, o, _), _ = lax.scan(body, (m, l, o, jnp.int32(0)), xs=None, length=nblocks, unroll=1)

    # Normalize; zeros for fully-masked rows
    out = jnp.where((l > 0)[..., None], o / l[..., None], 0.0).astype(out_dtype)  # (B,H,Lq,Dh)
    return jnp.transpose(out, (0, 2, 1, 3))  # (B,Lq,H,Dh)
