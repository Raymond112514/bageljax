from typing import Optional

import jax.numpy as jnp
import flax.linen as nn


class TokenEmbedder(nn.Module):
    vocab_size: int = 152_064
    hidden_dim: int = 3_584
    init_std: float = 0.02
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.normal(self.init_std, dtype=self.param_dtype),
            (self.vocab_size, self.hidden_dim),
        )

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.weight, token_ids, axis=0)


class LogitsHead(nn.Module):
    vocab_size: int = 152_064
    hidden_dim: int = 3_584
    init_std: float = 0.02
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.normal(self.init_std, dtype=self.param_dtype),
            (self.vocab_size, self.hidden_dim),
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.einsum("...d,vd->...v", hidden_states, self.weight)
