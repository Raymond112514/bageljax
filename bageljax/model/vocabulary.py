from typing import Optional

import jax.numpy as jnp
import flax.linen as nn

from bageljax.utils.jax_utils import add_batch_sharding_constraint


class ActionProjector(nn.Module):
    """Projects continuous action vectors into one or more LLM token embeddings."""
    action_dim: int = 8  # 7D joint velocity + 1D binarized gripper
    hidden_dim: int = 3_584
    init_std: float = 0.02
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
        # action: (B, action_dim) or (B, T, action_dim), float32
        action = add_batch_sharding_constraint(action, where="input to action projector")
        if action.ndim == 2:
            action = action[:, None, :]
        x = action.astype(jnp.bfloat16)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.normal(self.init_std),
            param_dtype=self.param_dtype,
        )(x)                              # (B, T, hidden_dim)
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        return x

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
        token_ids = add_batch_sharding_constraint(token_ids, where="input to token embedder")
        return jnp.take(self.weight, token_ids, axis=0)


class LogitsHead(nn.Module):
    vocab_size: int = 512 # allows for a maximum of 512 unique distances. You don't have to use the full vocab
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
        hidden_states = add_batch_sharding_constraint(hidden_states, where="input to logits head")
        return jnp.einsum("...d,vd->...v", hidden_states, self.weight)
