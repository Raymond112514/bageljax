from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import einops  # type: ignore
import jax
import jax.numpy as jnp
from flax import linen as nn

# -----------------------------------------------------------------------------
# Utility functions & layers
# -----------------------------------------------------------------------------

def swish(x: jnp.ndarray) -> jnp.ndarray:  # alias for SiLU
    return jax.nn.silu(x)


class AttnBlock(nn.Module):
    """Single‑head self‑attention in `HW` spatial grid, residual style."""

    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # (B, H, W, C)
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)

        q = nn.Conv(self.channels, kernel_size=(1, 1))(h)
        k = nn.Conv(self.channels, kernel_size=(1, 1))(h)
        v = nn.Conv(self.channels, kernel_size=(1, 1))(h)

        # (B, 1, HW, C)
        q = einops.rearrange(q, "b h w c -> b 1 (h w) c")
        k = einops.rearrange(k, "b h w c -> b 1 (h w) c")
        v = einops.rearrange(v, "b h w c -> b 1 (h w) c")

        scale = 1.0 / jnp.sqrt(self.channels)
        attn_weights = jnp.einsum("b n q c, b n k c -> b n q k", q, k) * scale
        attn = jax.nn.softmax(attn_weights, axis=-1)
        h_ = jnp.einsum("b n q k, b n k c -> b n q c", attn, v)
        h_ = einops.rearrange(h_, "b 1 (h w) c -> b h w c", h=x.shape[1])

        h_ = nn.Conv(self.channels, kernel_size=(1, 1))(h_)
        return x + h_


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        h = x
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding='SAME')(h)

        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding='SAME')(h)

        # <-- instantiate shortcut **only** if needed
        if self.in_channels != self.out_channels:
            x = nn.Conv(self.out_channels, (1, 1))(x)

        return x + h


class Downsample(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # (B, H, W, C)
        # Same spatial handling as the PyTorch asymmetric pad → conv‑stride‑2.
        pad = ((0, 1), (0, 1))  # pad H and W at "end" by +1.
        x = jnp.pad(x, pad_width=((0, 0),) + pad + ((0, 0),), mode="constant")
        return nn.Conv(self.channels, kernel_size=(3, 3), strides=(2, 2), padding="VALID")(x)


class Upsample(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # (B, H, W, C)
        x = jnp.repeat(x, 2, axis=1)
        x = jnp.repeat(x, 2, axis=2)
        return nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)


# -----------------------------------------------------------------------------
# Encoder / Decoder
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    resolution: int
    in_channels: int
    ch: int
    ch_mult: Sequence[int]
    num_res_blocks: int
    z_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # (B, H, W, C)
        h = nn.Conv(self.ch, kernel_size=(3, 3), padding="SAME")(x)
        curr_res = self.resolution
        block_in = self.ch

        # Downsampling hierarchy
        for i_level, mult in enumerate(self.ch_mult):
            block_out = self.ch * mult
            for _ in range(self.num_res_blocks):
                h = ResnetBlock(block_in, block_out)(h)
                block_in = block_out
            if i_level != len(self.ch_mult) - 1:
                h = Downsample(block_in)(h)
                curr_res //= 2

        # Middle (residual‑attn‑residual)
        h = ResnetBlock(block_in, block_in)(h)
        h = AttnBlock(block_in)(h)
        h = ResnetBlock(block_in, block_in)(h)

        # Output projection to (mean, logvar)
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(h)
        h = swish(h)
        h = nn.Conv(self.z_channels * 2, kernel_size=(3, 3), padding="SAME")(h)
        return h


class Decoder(nn.Module):
    resolution: int
    out_ch: int
    ch: int
    ch_mult: Sequence[int]
    num_res_blocks: int
    z_channels: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:  # (B, h, w, C)
        # Initial projection from latent z
        block_in = self.ch * self.ch_mult[-1]
        h = nn.Conv(block_in, kernel_size=(3, 3), padding="SAME")(z)

        # Middle
        h = ResnetBlock(block_in, block_in)(h)
        h = AttnBlock(block_in)(h)
        h = ResnetBlock(block_in, block_in)(h)

        # Upsampling hierarchy (reverse order)
        for i_level, mult in list(enumerate(self.ch_mult))[::-1]:
            block_out = self.ch * mult
            for _ in range(self.num_res_blocks + 1):
                h = ResnetBlock(block_in, block_out)(h)
                block_in = block_out
            if i_level != 0:
                h = Upsample(block_in)(h)

        # Final conversion to RGB / logits
        h = nn.GroupNorm(num_groups=32, epsilon=1e-6)(h)
        h = swish(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
        return h


# -----------------------------------------------------------------------------
# Latent distribution helper
# -----------------------------------------------------------------------------

class DiagonalGaussian(nn.Module):
    sample: bool = True
    chunk_axis: int = -1  # channel‑last → split on channels

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        mean, logvar = jnp.split(z, 2, axis=self.chunk_axis)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            rng = self.make_rng("gaussian")
            eps = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
            return mean + std * eps
        return mean


# -----------------------------------------------------------------------------
# High‑level AutoEncoder
# -----------------------------------------------------------------------------

@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159


class AutoEncoder(nn.Module):
    params: AutoEncoderParams
    sample_latent: bool = True  # expose for flexibility

    def setup(self):
        p = self.params
        self.encoder = Encoder(
            resolution=p.resolution,
            in_channels=p.in_channels,
            ch=p.ch,
            ch_mult=p.ch_mult,
            num_res_blocks=p.num_res_blocks,
            z_channels=p.z_channels,
        )
        self.decoder = Decoder(
            resolution=p.resolution,
            out_ch=p.out_ch,
            ch=p.ch,
            ch_mult=p.ch_mult,
            num_res_blocks=p.num_res_blocks,
            z_channels=p.z_channels,
        )
        self.reg = DiagonalGaussian(sample=self.sample_latent)

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self.reg(self.encoder(x))
        return self.params.scale_factor * (z - self.params.shift_factor)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        z = z / self.params.scale_factor + self.params.shift_factor
        return self.decoder(z)

    # ------------------------------------------------------------------
    # forward / __call__
    # ------------------------------------------------------------------

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.decode(self.encode(x))


# -----------------------------------------------------------------------------
# Convenience factory (weights need separate conversion step)
# -----------------------------------------------------------------------------

def build_autoencoder(sample_latent: bool = True) -> AutoEncoder:
    """Returns the initialized Flax `AutoEncoder` module and its params dict.

    Example usage:
        >>> rng = jax.random.PRNGKey(0)
        >>> ae = build_autoencoder()
        >>> params = ae.init(rng, jnp.zeros((1, 256, 256, 3)))
        >>> out = ae.apply(params, jnp.zeros((1, 256, 256, 3)), rngs={"gaussian": rng})
    """

    model = AutoEncoder(AutoEncoderParams(), sample_latent=sample_latent)
    return model
