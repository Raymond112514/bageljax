import os
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from typing import Any, Callable, Optional, Tuple
from flax.training import checkpoints
from flax.training.train_state import TrainState
import numpy as np
import optax
import torch
from safetensors.torch import load_file as load_sft
from copy import deepcopy
import functools

from bageljax.vocabulary import TokenEmbedder, LogitsHead
from bageljax.vision_encoder import VisionEncoder
from bageljax.vae2llm2vae import TimeEmbedder, VAE2LLM, LLM2VAE
from bageljax.mixture_of_transformers import MixtureOfTransformers
from bageljax.common import ModuleDict
from bageljax.autoencoder import build_autoencoder
from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

################################################
# Hyperparameters
################################################
SEED = 0
TEXT_2_IMG_MAX_SEQ_LEN = 4350 # max image tokens is 64x64 + 2, a reasonable max prompt tokens is 250 + 2
PAD_TOKEN_ID = 0 # it doesn't matter what this is

# We will be saving checkpoints in this script, and we want to avoid using orbax
# Prevent flax.checkpoints from using Orbax backend
flax.config.update('flax_use_orbax_checkpointing', False)

################################################
# Initialize the main Bagel model
################################################
networks = {
    "token_embedder": TokenEmbedder(),
    "vision_encoder": VisionEncoder(),
    "time_embedder": TimeEmbedder(),
    "vae2llm": VAE2LLM(),
    "llm2vae": LLM2VAE(),
    "mixture_of_transformers": MixtureOfTransformers(),
    "logits_head": LogitsHead(),
}

model_def = ModuleDict(networks)

def init_fn(rng):
    rng, init_rng = jax.random.split(rng)

    # For init, let's pick reasonable values of some of the input parameters
    B = 1 # batch size
    H, W = 672, 672 # image height and width, 672 is divisible by 14 and 16
    vae_latent_dim = 16 # VAE latent dimension
    llm_hidden_dim = 3584 # LLM hidden dimension
    L = 42*42 # sequence length, for llm2vae needs to be (H/16)*(W/16)

    params = model_def.init({'params': init_rng},
                                token_embedder = [
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                ],
                                vision_encoder = [
                                    jnp.zeros((B, H, W, 3), dtype=jnp.uint8),
                                ],
                                time_embedder = [
                                    jnp.zeros((B,), dtype=jnp.float32),
                                ],
                                vae2llm = [
                                    jnp.zeros((B, H // 8, W // 8, vae_latent_dim), dtype=jnp.bfloat16),
                                ],
                                llm2vae = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                    (H // 16, W // 16),
                                ],
                                mixture_of_transformers = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                    jnp.zeros((B, 1, L, L), dtype=jnp.bfloat16),
                                ],
                                logits_head = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
                                ],
                            )["params"]
    train_state = TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        tx=optax.identity(),
    )
    return train_state

rng = jax.random.PRNGKey(SEED)
rng, key = jax.random.split(rng)
train_state = jax.jit(init_fn)(key)
print("Model initialized.")

# Load the checkpoint
checkpoint_path = "pretrained_weights/bagel"
train_state = checkpoints.restore_checkpoint(checkpoint_path, target=train_state)
print("Pre-trained weights loaded.")

################################################
# Load the autoencoder
################################################
def make_jitted_encode(ae):
    """Return a jit-compiled encode(X) -> z."""
    @functools.partial(jax.jit,
                       static_argnames=("method",))
    def _encode(variables, x, key, *, method):
        return ae.apply(variables, x,
                        rngs={"gaussian": key},
                        method=method)
    # bind the method so callers don’t pass it
    return functools.partial(_encode, method=ae.encode)


def make_jitted_decode(ae):
    """Return a jit-compiled decode(z) -> X_rec."""
    @functools.partial(jax.jit,
                       static_argnames=("method",))
    def _decode(variables, z, *, method):
        return ae.apply(variables, z, method=method)
    return functools.partial(_decode, method=ae.decode)

ae = build_autoencoder(sample_latent=True)
rng, key = jax.random.split(rng)
ae_variables = ae.init(key, jnp.zeros((1, 256, 256, 3)))
print("Autoencoder initialized.")

# Load weights for the autoencoder
ae_checkpoint_path = "pretrained_weights/ae"
ae_variables = checkpoints.restore_checkpoint(ae_checkpoint_path, target=ae_variables)
print("Autoencoder weights loaded.")

ae_encode = make_jitted_encode(ae)
ae_decode = make_jitted_decode(ae)

################################################
# Tokenizer
################################################
tokenizer_load_path = "pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

################################################
# Text -> Image inference function
################################################
def text2image(prompt: str, image_shape: Tuple[int, int]=(1024, 1024)):
    # We need to prepare two max length arrays, one with text+vae, the other with just vae, allowing us to do CFG
    # To make CFG easy, we will left-pad

    # The first thing we will do is tokenize the prompt, and the bos and eos tokens, then left pad to max-length - num_vae_tokens
    # Then we will pass this, alongside the desired image shape (this will be a static parameter) and an rng to a jax.jit wrapped function
    # which will run the image generation process with batch size 1.
    text_ids = tokenizer.encode(prompt)
    text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]

    num_vae_tokens = (image_shape[0] // 16) * (image_shape[1] // 16) + 2
    text_padding_size = TEXT_2_IMG_MAX_SEQ_LEN - num_vae_tokens - len(text_ids)
    non_padding_text_size = len(text_ids)
    text_ids = [PAD_TOKEN_ID] * text_padding_size + text_ids

    text_ids = jnp.array(text_ids, dtype=jnp.int32)
    text_ids_mask = jnp.concatenate([jnp.zeros((text_padding_size,), dtype=bool), jnp.ones((non_padding_text_size,), dtype=bool)])
    text_rope_ids = jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), jnp.arange(non_padding_text_size, dtype=jnp.int32)])

    rng, key = jax.random.split(rng) # update global rng

    # So far no batch dimension has been introduced

    @partial(jax.jit, static_argnames=("image_shape",))
    def generate_image(train_state, ae_variables, token_types, text_ids, text_ids_mask, text_rope_ids, rng, image_shape):
        # none of the function inputs have a batch dimension

        # First we'll embed the text tokens
        text_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=text_ids[None, :],
            train=False,
            name="token_embedder",
        )
        # text_embeds now has a batch dimension

        # Next, we'll sample the VAE latents
        rng, key = jax.random.split(rng)
        x = jax.random.normal(key, (1, image_shape[0] // 8, image_shape[1] // 8, 16), dtype=jnp.bfloat16)
        # x has a batch dimension

        # Sample time in float32
        denoising_timesteps = jnp.linspace(1.0, 0.0, num=50, dtype=jnp.float32)
        timestep_shift = 3.0
        denoising_timesteps = timestep_shift * denoising_timesteps / (1 + (timestep_shift - 1) * denoising_timesteps)
        dts = denoising_timesteps[:-1] - denoising_timesteps[1:]
        dts = dts[None, ...]
        # dts has a batch dimension

        # Let's embed the start and end image tokens
        image_special_token_ids = jnp.array([new_token_ids['start_of_image'], new_token_ids['end_of_image']], dtype=jnp.int32)
        image_special_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=image_special_token_ids[None, :],
            train=False,
            name="token_embedder",
        )
        # image_special_embeds has a batch dimension

        # Next, prepare the full sequence attention masks, rope ids, and token types
        num_vae_tokens = (image_shape[0] // 16) * (image_shape[1] // 16) + 2
        full_seq_rope_ids = jnp.concatenate([text_rope_ids, jnp.ones((num_vae_tokens,), dtype=jnp.int32) * (text_rope_ids[-1] + 1)], axis=0)[None, :]
        full_seq_token_types = token_types[None, :]  # add batch dimension
    
        # Attn bias
        # We will construct an array like token_types, but storing 0 for pad tokens, 1 for causal tokens, and 2 for non-causal tokens
        # It's different because text special tokens in the vae portion are non-causal
        attention_token_types = jnp.concatenate([token_types[:-num_vae_tokens], 2*jnp.ones((num_vae_tokens,), dtype=jnp.int32)], axis=0)[None, :]
        L = TEXT_2_IMG_MAX_SEQ_LEN
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        is_vae = (attention_token_types == 2)
        # allow VAE queries to see all VAE keys
        allowed = jnp.where(is_vae[:, :, None],
                            causal[None],              # broadcast B
                            causal[None])
        allowed = jnp.where(is_vae[:, :, None] & is_vae[:, None, :],
                    True,
                    allowed)
        # padding (attention_token_types==0) sees nothing, and is seen by nothing
        allowed = jnp.where((attention_token_types[:, :, None] == 0) | (attention_token_types[:, None, :] == 0), False, allowed)

        attn_bias = jnp.where(allowed, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)

        # there will be a total of len(dts) == 49 denoising steps
        def step_fn(carry, _):
            x_t, denoising_t, dts_idx = carry
            # all three are expected to have a batch dimension

            # Ok, now we need to call VAE2LLM
            x_t_embeds, _ = train_state.apply_fn(
                {"params": train_state.params},
                z16=x_t,
                train=False,
                name="vae2llm",
            )

            # Concat the special text tokens
            vae_seq = jnp.concatenate([image_special_embeds[:, 0:1, :], x_t_embeds, image_special_embeds[:, 1:2, :]], axis=1)

            # Concat the text and vae embeddings
            full_seq = jnp.concatenate([text_embeds, vae_seq], axis=1)

            # Run the mixture of transformers
            hidden_states = train_state.apply_fn(
                {"params": train_state.params},
                x=full_seq,
                token_types=full_seq_token_types,
                rope_pos_ids=full_seq_rope_ids,
                attn_bias=attn_bias,
                train=False,
                name="mixture_of_transformers",
            )

            # Now we need to extract just the vae image tokens
            vae_hidden_states = hidden_states[:, -num_vae_tokens:, :]
            vae_hidden_states = vae_hidden_states[:, 1:-1, :]  # remove start and end special tokens

            # Project with LLM2VAE
            v_pred = train_state.apply_fn(
                {"params": train_state.params},
                tokens=vae_hidden_states,
                grid_hw=(image_shape[0] // 16, image_shape[1] // 16),
                train=False,
                name="llm2vae",
            )

            