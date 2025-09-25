import jax.numpy as jnp
import flax.linen as nn

class ProprioProjector(nn.Module):
    hidden_dim: int = 3584
    init_std: float = 0.02
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, proprio: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.truncated_normal(self.init_std),  # BERT-style
            bias_init=nn.initializers.zeros,
            name="fc1",
        )(proprio)
        x = nn.swish(x)
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.truncated_normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="fc2",
        )(x)
        return x