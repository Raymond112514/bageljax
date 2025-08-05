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
from copy import deepcopy
import functools
from functools import partial
from PIL import Image

from bageljax.vocabulary import TokenEmbedder, LogitsHead
from bageljax.vision_encoder import VisionEncoder
from bageljax.vae2llm2vae import TimeEmbedder, VAE2LLM, LLM2VAE, group_2x2, ungroup_2x2
from bageljax.mixture_of_transformers import MixtureOfTransformers
from bageljax.common import ModuleDict
from bageljax.autoencoder import build_autoencoder
from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

# Set up jax compilation cache
jax.config.update("jax_compilation_cache_dir", "/home/pranav/.jax_compilation_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

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
    global rng # use the global rng

    # We need to prepare two max length arrays, one with text+vae, the other with just vae, allowing us to do CFG
    # To make CFG easy, we will left-pad

    text_ids = tokenizer.encode(prompt)
    text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]

    num_vae_tokens = (image_shape[0] // 16) * (image_shape[1] // 16) + 2
    text_padding_size = TEXT_2_IMG_MAX_SEQ_LEN - num_vae_tokens - len(text_ids)
    non_padding_text_size = len(text_ids)
    text_ids = [PAD_TOKEN_ID] * text_padding_size + text_ids
    cfg_text_ids = [PAD_TOKEN_ID] * (text_padding_size + non_padding_text_size)

    text_ids = jnp.array(text_ids, dtype=jnp.int32)
    text_rope_ids = jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), jnp.arange(non_padding_text_size, dtype=jnp.int32)])

    cfg_text_ids = jnp.array(cfg_text_ids, dtype=jnp.int32)
    cfg_text_rope_ids = jnp.zeros((len(cfg_text_ids),), dtype=jnp.int32)

    token_types = jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), jnp.ones((non_padding_text_size,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32), 2*jnp.ones((num_vae_tokens-2,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)])
    cfg_token_types = jnp.concatenate([jnp.zeros((text_padding_size+non_padding_text_size,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32), 2*jnp.ones((num_vae_tokens-2,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)])

    rng, key = jax.random.split(rng) # use global rng

    # So far no batch dimension has been introduced

    @partial(jax.jit, static_argnames=("image_shape",))
    def generate_image(train_state, token_types, cfg_token_types, text_ids, text_rope_ids, cfg_text_ids, cfg_text_rope_ids, rng, image_shape):
        # combine text_ids and cfg_text_ids on the batch axis so all computation can be batch parallelized
        all_text_ids = jnp.stack([text_ids, cfg_text_ids])

        all_text_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=all_text_ids,
            name="token_embedder",
        )
        text_embeds = all_text_embeds[0:1] # we'll need this for the CFG-free denoising steps

        # Next, we'll sample the VAE latents
        rng, key = jax.random.split(rng)
        x = jax.random.normal(key, (1, image_shape[0] // 8, image_shape[1] // 8, 16), dtype=jnp.bfloat16)

        # Sample time in float32
        denoising_timesteps = jnp.linspace(1.0, 0.0, num=50, dtype=jnp.float32)
        timestep_shift = 3.0
        denoising_timesteps = timestep_shift * denoising_timesteps / (1 + (timestep_shift - 1) * denoising_timesteps)
        dts = denoising_timesteps[:-1] - denoising_timesteps[1:]
        dts = dts.astype(jnp.bfloat16) # dts should be in bfloat16, denoising_timesteps in float32

        # In the PyTorch code base, CFG was only applied when timestep was between 0.4 and 1. 
        # Based on the (static) value of the time-step shifted denoising_timesteps above, 
        # this means that we do 41 steps with CFG, and 8 steps without
        # this is a total of 49 denoising steps, which equals len(dts)

        # Let's embed the start and end image tokens
        image_special_token_ids = jnp.array([new_token_ids['start_of_image'], new_token_ids['end_of_image']], dtype=jnp.int32)
        image_special_token_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=image_special_token_ids[None, :],
            name="token_embedder",
        )

        # Next, prepare the full sequence attention masks, rope ids, and token types
        num_vae_tokens = (image_shape[0] // 16) * (image_shape[1] // 16) + 2
        full_seq_rope_ids = jnp.concatenate([text_rope_ids, jnp.ones((num_vae_tokens,), dtype=jnp.int32) * (text_rope_ids[-1] + 1)], axis=0)[None, :]
        cfg_full_seq_rope_ids = jnp.concatenate([cfg_text_rope_ids, jnp.zeros((num_vae_tokens,), dtype=jnp.int32)])[None, :]
        all_full_seq_rope_ids = jnp.concatenate([full_seq_rope_ids, cfg_full_seq_rope_ids], axis=0)

        full_seq_token_types = token_types[None, :]  # add batch dimension
        cfg_full_seq_token_types = cfg_token_types[None, :]
        all_full_seq_token_types = jnp.concatenate([full_seq_token_types, cfg_full_seq_token_types], axis=0)
    
        # Now we'll construct the attention bias
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
        allowed_attention = jnp.where((attention_token_types[:, :, None] == 0) | (attention_token_types[:, None, :] == 0), False, allowed)
        full_seq_attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)

        # Create an attn_bias for the CFG sequence as well. We can reuse a lot of the components
        cfg_attention_token_types = jnp.concatenate([cfg_token_types[:-num_vae_tokens], 2*jnp.ones((num_vae_tokens,), dtype=jnp.int32)], axis=0)[None, :]
        # padding (cfg_attention_token_types==0) sees nothing, and is seen by nothing
        cfg_allowed_attention = jnp.where((cfg_attention_token_types[:, :, None] == 0) | (cfg_attention_token_types[:, None, :] == 0), False, allowed)
        cfg_full_seq_attn_bias = jnp.where(cfg_allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        
        all_full_seq_attn_bias = jnp.concatenate([full_seq_attn_bias, cfg_full_seq_attn_bias], axis=0)

        def step_fn_cfg(carry, _):
            x_t, denoising_t, dts_idx = carry["x_t"], carry["denoising_t"], carry["dts_idx"]
            # we expect x_t and denoising_t to have a batch dimension, and dts_idx to not (note the batch dimension of denoising_t is its only dimension)

            # Ok, now we need to call VAE2LLM
            z64 = group_2x2(x_t)
            x_t_embeds, _ = train_state.apply_fn(
                {"params": train_state.params},
                z64=z64,
                name="vae2llm",
            )

            # Embed time and add to x_t_embeds
            time_embedding = train_state.apply_fn(
                {"params": train_state.params},
                timesteps=denoising_t,
                name="time_embedder",
            )
            x_t_embeds = x_t_embeds + time_embedding[:, None, :] # make sure to introduce sequence dimension into time_embedding

            # Concat the special text tokens
            vae_seq = jnp.concatenate([image_special_token_embeds[:, 0:1, :], x_t_embeds, image_special_token_embeds[:, 1:2, :]], axis=1)

            # Concat the text and vae embeddings
            # text_embeds, which we will concatenate next line, has batch size 2 (for no cfg and w/ cfg)
            vae_seq = jnp.tile(vae_seq, (2, 1, 1))
            full_seq = jnp.concatenate([all_text_embeds, vae_seq], axis=1)

            # Run the mixture of transformers
            hidden_states = train_state.apply_fn(
                {"params": train_state.params},
                x=full_seq,
                token_types=all_full_seq_token_types,
                rope_pos_ids=all_full_seq_rope_ids,
                attn_bias=all_full_seq_attn_bias,
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
                name="llm2vae",
            )
            v_pred, uncond_v_pred = v_pred[0:1], v_pred[1:2]
            # these will have shape (B, H, W, 64)

            # CFG
            v = uncond_v_pred + 4.0 * (v_pred - uncond_v_pred)
            # global norms
            pre_cfg_norm = jnp.linalg.norm(v_pred)
            post_cfg_norm = jnp.linalg.norm(v)
            scale = jnp.clip(pre_cfg_norm / (post_cfg_norm + 1e-8), min=0.0, max=1.0)
            v = v * scale

            # 2x2 unpatchify v to make it the same shape as x
            v = ungroup_2x2(v)

            # Euler integration
            dt = jnp.take(dts, dts_idx)
            x_t = x_t - v * dt

            return {"x_t": x_t, "denoising_t": denoising_t - dt, "dts_idx": dts_idx + 1}, None

        # We will call this function 41 times (see explanation above)
        scan_result = jax.lax.scan(step_fn_cfg, {"x_t": x, "denoising_t": jnp.ones((1,), dtype=jnp.float32), "dts_idx": jnp.array([0], dtype=jnp.int32)}, xs=None, length=41)
        x, denoising_t, dts_idx = scan_result[0]["x_t"], scan_result[0]["denoising_t"], scan_result[0]["dts_idx"]

        def step_fn_no_cfg(carry, _):
            x_t, denoising_t, dts_idx = carry["x_t"], carry["denoising_t"], carry["dts_idx"]
            # we expect x_t and denoising_t to have a batch dimension, and dts_idx to not (note the batch dimension of denoising_t is its only dimension)

            # Ok, now we need to call VAE2LLM
            z64 = group_2x2(x_t)
            x_t_embeds, _ = train_state.apply_fn(
                {"params": train_state.params},
                z64=z64,
                name="vae2llm",
            )

            # Embed time and add to x_t_embeds
            time_embedding = train_state.apply_fn(
                {"params": train_state.params},
                timesteps=denoising_t,
                name="time_embedder",
            )
            x_t_embeds = x_t_embeds + time_embedding[:, None, :] # make sure to introduce sequence dimension into time_embedding

            # Concat the special text tokens
            vae_seq = jnp.concatenate([image_special_token_embeds[:, 0:1, :], x_t_embeds, image_special_token_embeds[:, 1:2, :]], axis=1)

            # Concat the text and vae embeddings
            full_seq = jnp.concatenate([text_embeds, vae_seq], axis=1)

            # Run the mixture of transformers
            hidden_states = train_state.apply_fn(
                {"params": train_state.params},
                x=full_seq,
                token_types=full_seq_token_types,
                rope_pos_ids=full_seq_rope_ids,
                attn_bias=full_seq_attn_bias,
                name="mixture_of_transformers",
            )

            # Now we need to extract just the vae image tokens
            vae_hidden_states = hidden_states[:, -num_vae_tokens:, :]
            vae_hidden_states = vae_hidden_states[:, 1:-1, :]  # remove start and end special tokens

            # Project with LLM2VAE
            v = train_state.apply_fn(
                {"params": train_state.params},
                tokens=vae_hidden_states,
                grid_hw=(image_shape[0] // 16, image_shape[1] // 16),
                name="llm2vae",
            )
            # this will have shape (B, H, W, 64)

            # 2x2 unpatchify v to make it the same shape as x
            v = ungroup_2x2(v)

            # Euler integration
            dt = jnp.take(dts, dts_idx)
            x_t = x_t - v * dt

            return {"x_t": x_t, "denoising_t": denoising_t - dt, "dts_idx": dts_idx + 1}, None

        # We will call this function 9 times
        scan_result = jax.lax.scan(step_fn_no_cfg, {"x_t": x, "denoising_t": denoising_t, "dts_idx": dts_idx}, xs=None, length=9)
        x = scan_result[0]["x_t"]

        return x # we don't feed through the vae decoder because that's been jitted separately

    gen_vae_latents = generate_image(train_state, token_types, cfg_token_types, text_ids, text_rope_ids, cfg_text_ids, cfg_text_rope_ids, key, image_shape)
    return gen_vae_latents.astype(jnp.float32) # The VAE operates in float32

prompt = "a green lantern"
gen_img_latent = text2image(prompt)
gen_img = ae_decode(ae_variables, gen_img_latent)
gen_img = np.array(gen_img)[0]
gen_img = np.clip((gen_img + 1) * 127.5, 0, 255).astype(np.uint8)
gen_img = Image.fromarray(gen_img)

gen_img.save("a_green_lantern.png")
