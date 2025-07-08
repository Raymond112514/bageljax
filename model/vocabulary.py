# vocabulary.py
# ---------------------------------------------------------------------
#  Two lightweight Flax layers that map between token-ids  ↔  hidden states
#  ─────────────────────────────────────────────────────────────────────
#  Shapes match the BAGEL checkpoint exactly:
#
#  ┌─────────────────────────────────────┬──────────────────────────────┐
#  │ PyTorch tensor                      │  Flax path in this file      │
#  ├─────────────────────────────────────┼──────────────────────────────┤
#  │ language_model.model.embed_tokens   │  TokenEmbedder/weight        │
#  │             .weight  (152 064,3584) │                              │
#  │ language_model.lm_head.weight       │  LogitsHead/weight           │
#  │                     (152 064,3584)  │                              │
#  └─────────────────────────────────────┴──────────────────────────────┘
# ---------------------------------------------------------------------
from typing import Optional

import jax.numpy as jnp
import flax.linen as nn


class TokenEmbedder(nn.Module):
    """
    Embeds integer token-ids into `hidden_dim` vectors.

    Parameters
    ----------
    vocab_size : int
        Number of entries in the vocabulary.
    hidden_dim : int
        Dimensionality of each token embedding.

    Returns
    -------
    emb : jnp.ndarray
        Shape `(…, hidden_dim)` – same leading shape as the input ids.
    """
    vocab_size: int = 152_064
    hidden_dim: int = 3_584
    init_std: float = 0.02            # same as HF default

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.normal(self.init_std),
            (self.vocab_size, self.hidden_dim),
        )

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        # jnp.take supports arbitrary leading dims on token_ids
        return jnp.take(self.weight, token_ids, axis=0)      # (..., hidden_dim)


class LogitsHead(nn.Module):
    """
    Projects hidden states back to vocabulary-logits.

    If you prefer tied weights, pass the embedder’s weights via
    `tied_weight` at call-time.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden_dim : int
        Dimensionality of the incoming hidden states.

    Returns
    -------
    logits : jnp.ndarray
        Shape `(…, vocab_size)` – same leading dims as the hidden states.
    """
    vocab_size: int = 152_064
    hidden_dim: int = 3_584
    init_std: float = 0.02

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.normal(self.init_std),
            (self.vocab_size, self.hidden_dim),     # (V, D)
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        tied_weight: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute logits = h · Wᵀ  with explicit einsum.

        `hidden_states`  : shape (..., hidden_dim)
        `tied_weight`    : optional weight to use instead of self.weight
                           (enables weight-tying with the embedder).
        """
        w = tied_weight if tied_weight is not None else self.weight   # (V,D)
        # einsum keeps arbitrary leading batch / seq dims
        return jnp.einsum("...d,vd->...v", hidden_states, w)
