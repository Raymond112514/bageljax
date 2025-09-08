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
IMAGE_EDITING_MAX_SEQ_LEN = 13342 # 250 for prompt, 64x64 for vae, another 64x64 for vae conditioning, 70x70 for vit conditioning
VQA_MAX_SEQ_LEN = 5500 # 70x70+2 for image, then ~500 token for prompt + generated text
PAD_TOKEN_ID = 0 # it doesn't matter what this is

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
                                    jnp.zeros((B, H, W, 3), dtype=jnp.uint8), # I think it's better practice to have code assume images are pre-normalized, todo: change
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
ae_variables = ae.init(key, jnp.zeros((1, 1024, 1024, 3)))
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
        x = jax.random.normal(key, (1, image_shape[0] // 8, image_shape[1] // 8, 16), dtype=jnp.float32) # To match the PyTorch impl, we do Euler integration in float32

        # Sample time in float32
        denoising_timesteps = jnp.linspace(1.0, 0.0, num=50, dtype=jnp.float32)
        timestep_shift = 3.0
        denoising_timesteps = timestep_shift * denoising_timesteps / (1 + (timestep_shift - 1) * denoising_timesteps)
        dts = denoising_timesteps[:-1] - denoising_timesteps[1:]

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
        full_seq_attn_bias = full_seq_attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol

        # Create an attn_bias for the CFG sequence as well. We can reuse a lot of the components
        cfg_attention_token_types = jnp.concatenate([cfg_token_types[:-num_vae_tokens], 2*jnp.ones((num_vae_tokens,), dtype=jnp.int32)], axis=0)[None, :]
        # padding (cfg_attention_token_types==0) sees nothing, and is seen by nothing
        cfg_allowed_attention = jnp.where((cfg_attention_token_types[:, :, None] == 0) | (cfg_attention_token_types[:, None, :] == 0), False, allowed)
        cfg_full_seq_attn_bias = jnp.where(cfg_allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        cfg_full_seq_attn_bias = cfg_full_seq_attn_bias.astype(jnp.bfloat16)
        
        all_full_seq_attn_bias = jnp.concatenate([full_seq_attn_bias, cfg_full_seq_attn_bias], axis=0)

        def step_fn_cfg(carry, _):
            x_t, denoising_t, dts_idx = carry["x_t"], carry["denoising_t"], carry["dts_idx"]
            # we expect x_t and denoising_t to have a batch dimension, and dts_idx to not (note the batch dimension of denoising_t is its only dimension)

            # Ok, now we need to call VAE2LLM
            z64 = group_2x2(x_t.astype(jnp.bfloat16))
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
            v_pred = ungroup_2x2(v_pred).astype(jnp.float32) # 2x2 unpatchify, convert to float32 which next few steps require
            v_pred, uncond_v_pred = v_pred[0:1], v_pred[1:2]

            # CFG
            v = uncond_v_pred + 4.0 * (v_pred - uncond_v_pred)
            # global norms
            pre_cfg_norm = jnp.linalg.norm(v_pred)
            post_cfg_norm = jnp.linalg.norm(v)
            scale = jnp.clip(pre_cfg_norm / (post_cfg_norm + 1e-8), min=0.0, max=1.0)
            v = v * scale

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
            z64 = group_2x2(x_t.astype(jnp.bfloat16))
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
            v = ungroup_2x2(v).astype(jnp.float32) # 2x2 unpatchify, convert to float32 which next few steps require

            # Euler integration
            dt = jnp.take(dts, dts_idx)
            x_t = x_t - v * dt

            return {"x_t": x_t, "denoising_t": denoising_t - dt, "dts_idx": dts_idx + 1}, None

        # We will call this function 8 times
        scan_result = jax.lax.scan(step_fn_no_cfg, {"x_t": x, "denoising_t": denoising_t, "dts_idx": dts_idx}, xs=None, length=8)
        x = scan_result[0]["x_t"]

        return x

    gen_vae_latents = generate_image(train_state, token_types, cfg_token_types, text_ids, text_rope_ids, cfg_text_ids, cfg_text_rope_ids, key, image_shape)
    return gen_vae_latents # already in float32

################################################
# Text + Image -> Image inference function
################################################
def image_editing(prompt: str, image: Image):
    global rng # use the global rng

    # We need to prepare three max length arrays, one with everything, one with just text, and one with just image, allowing us to do CFG
    # The order of the elements is vae context, vit context, text prompt, vae generation
    # Everything in the above except for text is of fixed size (i.e., we know the size). We have to pad text, so we could (1) keep text at the left 
    # but adjust rope IDs so that it acts as if it's in the middle, or (2) simply pad in the middle.
    # I think option (2) is cleaner.

    text_ids = tokenizer.encode(prompt)
    text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]

    # Resize the image. Rules: each dimension needs to be a multiple of 14, and needs to be <= 980 pixels so that it can fit through the ViT
    # each dimension also needs to be a multiple of 16, so that the VAE can process it. Other than these rules, we will also try to resize it 
    # trying to preserve aspect ratio as much as possible. We will use LANCZOS with anti-alias set to true for downsampling.
    BASE, MAX_SIDE = 112, 980  # 14 x 16 = 112
    w, h = image.size
    scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0)  # don't exceed 980; avoids unintended upscaling
    tw, th = int(w * scale), int(h * scale)
    tw, th = max(BASE, (tw // BASE) * BASE), max(BASE, (th // BASE) * BASE)  # snap down to 112-multiples
    tw, th = min(tw, MAX_SIDE - (MAX_SIDE % BASE)), min(th, MAX_SIDE - (MAX_SIDE % BASE))  # final clamp
    image = image.resize((tw, th), resample=Image.LANCZOS)  # LANCZOS automatcially does antialiased downsampling

    # Normalize and convert to tensor
    image = np.array(image, dtype=np.float32) / 127.5 - 1
    image_float32 = jnp.array(image, dtype=jnp.float32)
    image = jnp.array(image, dtype=jnp.bfloat16)
    image_shape = (image.shape[0], image.shape[1])

    # Pad the text ids, and also prepare useful token sequence length variables
    num_vae_tokens = (image_shape[0] // 16) * (image_shape[1] // 16) + 2
    num_vit_tokens = (image_shape[0] // 14) * (image_shape[1] // 14) + 2
    text_padding_size = IMAGE_EDITING_MAX_SEQ_LEN - 2 * num_vae_tokens - num_vit_tokens - len(text_ids)
    non_padding_text_size = len(text_ids)
    num_text_tokens = non_padding_text_size + text_padding_size
    text_ids = [PAD_TOKEN_ID] * text_padding_size + text_ids
    text_ids = jnp.array(text_ids, dtype=jnp.int32)
    
    # Next construct rope id arrays for all three sequences
    rope_ids = jnp.concatenate([
        0 * jnp.ones((num_vae_tokens,), dtype=jnp.int32),
        1 * jnp.ones((num_vit_tokens,), dtype=jnp.int32),
        jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), 2 + jnp.arange(non_padding_text_size, dtype=jnp.int32)]), # note: the rope id for padding doesn't matter
        (non_padding_text_size + 2) * jnp.ones((num_vae_tokens,), dtype=jnp.int32),
    ])
    cfg_text_rope_ids = jnp.concatenate([
        0 * jnp.ones((num_vae_tokens,), dtype=jnp.int32),
        1 * jnp.ones((num_vit_tokens,), dtype=jnp.int32),
        jnp.zeros((num_text_tokens,), dtype=jnp.int32), # note: the rope id for padding doesn't matter
        2 * jnp.ones((num_vae_tokens,), dtype=jnp.int32),
    ])
    cfg_image_rope_ids = jnp.concatenate([
        0 * jnp.ones((num_vae_tokens,), dtype=jnp.int32), # note: the rope id for padding doesn't matter
        0 * jnp.ones((num_vit_tokens,), dtype=jnp.int32), # note: the rope id for padding doesn't matter
        jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), jnp.arange(non_padding_text_size, dtype=jnp.int32)]), # note: the rope id for padding doesn't matter
        non_padding_text_size * jnp.ones((num_vae_tokens,), dtype=jnp.int32),
    ])
    
    # Next, construct the token types arrays for all three sequences. Remember, 0 is for padding, 1 is for text/vit, and 2 is for vae
    token_types = jnp.concatenate([
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), 2 * jnp.ones((num_vae_tokens-2), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)]),
        1 * jnp.ones((num_vit_tokens,), dtype=jnp.int32),
        jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), 1 * jnp.ones((non_padding_text_size,), dtype=jnp.int32)]),
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), 2 * jnp.ones((num_vae_tokens-2), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)]),
    ])
    cfg_text_token_types = jnp.concatenate([
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), 2 * jnp.ones((num_vae_tokens-2), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)]),
        1 * jnp.ones((num_vit_tokens,), dtype=jnp.int32),
        jnp.zeros((num_text_tokens,), dtype=jnp.int32),
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), 2 * jnp.ones((num_vae_tokens-2), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)]),
    ])
    cfg_image_token_types = jnp.concatenate([
        jnp.zeros((num_vae_tokens,), dtype=jnp.int32),
        jnp.zeros((num_vit_tokens,), dtype=jnp.int32),
        jnp.concatenate([jnp.zeros((text_padding_size,), dtype=jnp.int32), 1 * jnp.ones((non_padding_text_size,), dtype=jnp.int32)]),
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), 2 * jnp.ones((num_vae_tokens-2), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32)]),
    ])

    # The ViT can process the raw image, but the VAE (when used for image conditioning) needs VAE latents
    rng, key = jax.random.split(rng) # split from the global rng which we have captured
    image_latents = ae_encode(ae_variables, image_float32[None, ...], key)
    image_latents = image_latents[0] # remove batch dim for consistency with all other vars so far, which don't have a batch dim yet
    image_latents = image_latents.astype(jnp.bfloat16)

    rng, key = jax.random.split(rng) # use global rng for following call of generate_image

    # So far no batch dimension has been introduced

    @partial(jax.jit, static_argnames=("image_shape",))
    def generate_image(train_state, text_ids, image, image_latents, image_shape, token_types, cfg_text_token_types, cfg_image_token_types, rope_ids, cfg_text_rope_ids, cfg_image_rope_ids, rng):
        # Let's embed the start and end image tokens
        image_special_token_ids = jnp.array([new_token_ids['start_of_image'], new_token_ids['end_of_image']], dtype=jnp.int32)
        image_special_token_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=image_special_token_ids[None, :],
            name="token_embedder",
        )

        # Next, prepare part of token sequence corresponding to the VAE conditioning
        timestep_zero = jnp.zeros((1,), dtype=jnp.float32)
        timestep_zero_embedding = train_state.apply_fn(
            {"params": train_state.params},
            timesteps=timestep_zero,
            name="time_embedder",
        ) # will be of shape (1, hidden_dim), dtype bfloat16
        assert timestep_zero_embedding.ndim == 2
        pre_llm_image_latents = group_2x2(image_latents[None, ...]) # Will have shape (B, H // 16, W // 16, 64)
        pre_llm_image_latents, _ = train_state.apply_fn(
            {"params": train_state.params},
            z64=pre_llm_image_latents,
            name="vae2llm",
        )
        pre_llm_image_latents = pre_llm_image_latents + timestep_zero_embedding[:, None, :]
        pre_llm_image_latents = jnp.concatenate([image_special_token_embeds[:, 0:1], pre_llm_image_latents, image_special_token_embeds[:, 1:2]], axis=1) # add the special image tokens
        # Copy three times along batch dimension
        pre_llm_image_latents = jnp.concatenate([pre_llm_image_latents, pre_llm_image_latents, pre_llm_image_latents], axis=0)

        # Similarly, prepare part of token sequence corresponding to the ViT tokens
        pre_llm_vit_tokens = train_state.apply_fn(
            {"params": train_state.params},
            img=image[None, ...],
            name="vision_encoder",
        ) # will have shape (B, num_vit_tokens, llm_hidden_dim)
        pre_llm_vit_tokens = jnp.concatenate([image_special_token_embeds[:, 0:1], pre_llm_vit_tokens, image_special_token_embeds[:, 1:2]], axis=1) # add the special image tokens
        # Copy three times along batch dimension
        pre_llm_vit_tokens = jnp.concatenate([pre_llm_vit_tokens, pre_llm_vit_tokens, pre_llm_vit_tokens], axis=0)

        # Next, the text tokens
        text_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=text_ids[None, ...],
            name="token_embedder",
        )
        # Copy three times along batch dimension
        text_embeds = jnp.concatenate([text_embeds, text_embeds, text_embeds], axis=0)

        # Next, we'll sample the VAE latents
        rng, key = jax.random.split(rng)
        x = jax.random.normal(key, (1, image_shape[0] // 8, image_shape[1] // 8, 16), dtype=jnp.float32) # To match the PyTorch impl, we do Euler integration in float32

        # Sample time in float32
        denoising_timesteps = jnp.linspace(1.0, 0.0, num=50, dtype=jnp.float32)
        timestep_shift = 3.0
        denoising_timesteps = timestep_shift * denoising_timesteps / (1 + (timestep_shift - 1) * denoising_timesteps)
        dts = denoising_timesteps[:-1] - denoising_timesteps[1:]

        # In the PyTorch code base, CFG is applied for all timesteps. So 49 denoising steps, all with CFG
    
        # Now we'll construct the attention bias, which will be block-wise causal
        L = IMAGE_EDITING_MAX_SEQ_LEN
        # Start with a causal mask
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        num_vae_tokens = pre_llm_image_latents.shape[1]
        num_vit_tokens = pre_llm_vit_tokens.shape[1]
        num_text_tokens = text_ids.shape[0]
        # block-wise self-attention
        row1_mask = jnp.concatenate([jnp.ones((num_vae_tokens, num_vae_tokens), dtype=bool), jnp.zeros((num_vae_tokens, L-num_vae_tokens), dtype=bool)], axis=1)
        row2_mask = jnp.concatenate([jnp.ones((num_vit_tokens, num_vae_tokens + num_vit_tokens), dtype=bool), jnp.zeros((num_vit_tokens, L - num_vit_tokens - num_vae_tokens), dtype=bool)], axis=1)
        row3_mask = jnp.zeros((num_text_tokens, L), dtype=bool)
        row4_mask = jnp.ones((num_vae_tokens, L), dtype=bool)
        blockwise_mask = jnp.concatenate([row1_mask, row2_mask, row3_mask, row4_mask], axis=0)
        blockwise_causal_or_causal = blockwise_mask | causal
        # padding sees nothing, and is seen by nothing
        padding = token_types[None, :] == 0
        allowed_attention = jnp.where(padding[:, :, None] | padding[:, None, :], False, blockwise_causal_or_causal)
        attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        attn_bias = attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol

        # Create an attn_bias for the CFG sequences as well. We can reuse a lot of the components
        # Let's start with text CFG
        cfg_text_padding = cfg_text_token_types[None, :] == 0
        allowed_attention = jnp.where(cfg_text_padding[:, :, None] | cfg_text_padding[:, None, :], False, blockwise_causal_or_causal)
        cfg_text_attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        cfg_text_attn_bias = cfg_text_attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol

        # And image CFG
        cfg_image_padding = cfg_image_token_types[None, :] == 0
        allowed_attention = jnp.where(cfg_image_padding[:, :, None] | cfg_image_padding[:, None, :], False, blockwise_causal_or_causal)
        cfg_image_attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        cfg_image_attn_bias = cfg_image_attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol
        
        all_attn_biases = jnp.concatenate([attn_bias, cfg_text_attn_bias, cfg_image_attn_bias], axis=0)

        # Also concatenate rope ids and token types along the batch dimension for the 3 sequences
        all_rope_ids = jnp.stack([rope_ids, cfg_text_rope_ids, cfg_image_rope_ids], axis=0)
        all_token_types = jnp.stack([token_types, cfg_text_token_types, cfg_image_token_types], axis=0)

        def step_fn_cfg(carry, _):
            x_t, denoising_t, dts_idx = carry["x_t"], carry["denoising_t"], carry["dts_idx"]
            # we expect x_t and denoising_t to have a batch dimension, and dts_idx to not (note the batch dimension of denoising_t is its only dimension)

            # Ok, now we need to call VAE2LLM
            z64 = group_2x2(x_t.astype(jnp.bfloat16))
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

            # Repeat 3 times along the batch dimension
            vae_seq = jnp.concatenate([vae_seq, vae_seq, vae_seq], axis=0)

            # Arrange the full sequence
            full_seq = jnp.concatenate([pre_llm_image_latents, pre_llm_vit_tokens, text_embeds, vae_seq], axis=1)
            assert full_seq.dtype == jnp.bfloat16 # just to make sure

            # Run the mixture of transformers
            hidden_states = train_state.apply_fn(
                {"params": train_state.params},
                x=full_seq,
                token_types=all_token_types,
                rope_pos_ids=all_rope_ids,
                attn_bias=all_attn_biases,
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
            v_pred = v_pred.astype(jnp.float32) # convert to float32 which next few steps require
            v_pred, cfg_text_v_pred, cfg_image_v_pred = v_pred[0:1], v_pred[1:2], v_pred[2:3]

            # Text CFG
            v_cfg_text_applied = cfg_text_v_pred + 4.0 * (v_pred - cfg_text_v_pred)
            # we do the so called "text_channel" norm from the PyTorch codebase
            norm_v_pred = jnp.linalg.norm(v_pred, axis=-1, keepdims=True)
            norm_v_cfg_text_applied = jnp.linalg.norm(v_cfg_text_applied, axis=-1, keepdims=True)
            scale = jnp.clip(norm_v_pred / (norm_v_cfg_text_applied + 1e-8), min=0.0, max=1.0)
            v_cfg_text_applied = v_cfg_text_applied * scale

            # Image CFG
            v = cfg_image_v_pred + 2.0 * (v_cfg_text_applied - cfg_image_v_pred)
            # no normalization for image CFG

            # Unpatchify
            v = ungroup_2x2(v)

            # Euler integration
            dt = jnp.take(dts, dts_idx)
            x_t = x_t - v * dt

            return {"x_t": x_t, "denoising_t": denoising_t - dt, "dts_idx": dts_idx + 1}, None

        # We will call this function 49 times
        scan_result = jax.lax.scan(step_fn_cfg, {"x_t": x, "denoising_t": jnp.ones((1,), dtype=jnp.float32), "dts_idx": jnp.array([0], dtype=jnp.int32)}, xs=None, length=49)
        x = scan_result[0]["x_t"]

        return x


    gen_vae_latents = generate_image(train_state, text_ids, image, image_latents, image_shape, token_types, cfg_text_token_types, cfg_image_token_types, rope_ids, cfg_text_rope_ids, cfg_image_rope_ids, key)
    return gen_vae_latents # already in float32

################################################
# Text + Image -> Text inference function
################################################
def vqa(prompt: str, image: Image):
    # Resize and normalize the image
    BASE, MAX_SIDE = 14, 980
    w, h = image.size
    scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0)  # don't exceed 980; avoids unintended upscaling
    tw, th = int(w * scale), int(h * scale)
    tw, th = max(BASE, (tw // BASE) * BASE), max(BASE, (th // BASE) * BASE)  # snap down to 14-multiples
    tw, th = min(tw, MAX_SIDE - (MAX_SIDE % BASE)), min(th, MAX_SIDE - (MAX_SIDE % BASE))  # final clamp
    image = image.resize((tw, th), resample=Image.LANCZOS)  # LANCZOS automatcially does antialiased downsampling

    # Normalize and convert to tensor
    image = np.array(image, dtype=np.float32) / 127.5 - 1
    image = jnp.array(image, dtype=jnp.bfloat16)

    @jax.jit
    def embed_image_with_vit(train_state, image):
        # Pass through ViT
        pre_llm_vit_tokens = train_state.apply_fn(
            {"params": train_state.params},
            img=image[None, ...],
            name="vision_encoder",
        )

        # Let's embed the start and end image tokens
        image_special_token_ids = jnp.array([new_token_ids['start_of_image'], new_token_ids['end_of_image']], dtype=jnp.int32)
        image_special_token_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=image_special_token_ids[None, :],
            name="token_embedder",
        )

        # add the special image tokens
        pre_llm_vit_tokens = jnp.concatenate([image_special_token_embeds[:, 0:1], pre_llm_vit_tokens, image_special_token_embeds[:, 1:2]], axis=1)

        return pre_llm_vit_tokens

    pre_llm_vit_tokens = embed_image_with_vit(train_state, image)

    # With text generation, it's easier to pad on the right
    prompt_text_ids = tokenizer.encode(prompt)
    prompt_text_ids = [new_token_ids['bos_token_id']] + prompt_text_ids + [new_token_ids['eos_token_id']]
    # record the length so far, so we know where to split later
    num_tokens_in_prompt = len(prompt_text_ids)
    # add in the bos token to create running_text_ids
    running_text_ids = prompt_text_ids + [new_token_ids['bos_token_id']]
    curr_generation_idx = num_tokens_in_prompt
    # add in the pad tokens
    running_text_ids = running_text_ids + [PAD_TOKEN_ID] * (VQA_MAX_SEQ_LEN - len(running_text_ids) - pre_llm_vit_tokens.shape[1])

    # single token generation function
    @jax.jit
    def greedily_predict_next_token(train_state, pre_llm_vit_tokens, text_ids, curr_generation_idx, token_types, rope_pos_ids, attn_bias):
        # embed the text tokens
        text_embeds = train_state.apply_fn(
            {"params": train_state.params},
            token_ids=text_ids[None, ...],
            name="token_embedder",
        )

        # Concat with vit tokens
        full_seq = jnp.concatenate([pre_llm_vit_tokens, text_embeds], axis=1)

        # Feed into transformer
        hidden_states = train_state.apply_fn(
            {"params": train_state.params},
            x=full_seq,
            token_types=token_types[None, ...],
            rope_pos_ids=rope_pos_ids[None, ...],
            attn_bias=attn_bias[None, ...],
            name="mixture_of_transformers",
        )

        # Select the token we just generated
        curr_generation_idx = curr_generation_idx + pre_llm_vit_tokens.shape[1] # the curr_generation_idx is w.r.t. text tokens only so far
        generated_token_hidden = jnp.take(hidden_states, curr_generation_idx, axis=1)
        assert generated_token_hidden.ndim == 2
        # this should have shape (B, H). Add back in the sequence dimension
        generated_token_hidden = generated_token_hidden[:, None, :]

        # Feed into LLM head
        token_prediction_logits = train_state.apply_fn(
            {"params": train_state.params},
            hidden_states=generated_token_hidden,
            name="logits_head",
        ) # should have shape (B, 1, V)
        assert token_prediction_logits.shape == (1, 1, 152_064)
        token_prediction_logits = jnp.squeeze(token_prediction_logits)

        # Take the argmax
        greedy_token_id = jnp.argmax(token_prediction_logits)

        return greedy_token_id

    eos_predicted = False
    while not eos_predicted: # note, we assume the model will always eventually predict the eos token
        # Prepare token_types
        token_types = jnp.concatenate([
            jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=jnp.int32),
            jnp.ones((curr_generation_idx + 1), dtype=jnp.int32),
            jnp.zeros((len(running_text_ids) - curr_generation_idx - 1), dtype=jnp.int32),
        ])

        # Prepare rope ids
        rope_pos_ids = jnp.concatenate([
            jnp.zeros((pre_llm_vit_tokens.shape[1],), dtype=jnp.int32),
            1 + jnp.arange(len(running_text_ids), dtype=jnp.int32), # no harm in setting rope pos ids for pad tokens
        ])

        # Prepare attention mask
        # Start with a causal mask
        L = VQA_MAX_SEQ_LEN
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        # Enable full self-attention for vit tokens
        arr = jnp.concatenate([jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=bool), jnp.zeros((L - pre_llm_vit_tokens.shape[1]), dtype=bool)])
        self_attention_mask = jnp.matmul(arr[:, None], arr[None, :])
        block_causal = causal | self_attention_mask
        # padding sees nothing, and is seen by nothing
        padding = token_types[None, :] == 0
        allowed_attention = jnp.where(padding[:, :, None] | padding[:, None, :], False, block_causal)
        attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
        attn_bias = attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol

        # Predict next token
        next_token_id = greedily_predict_next_token(train_state, pre_llm_vit_tokens, jnp.array(running_text_ids, dtype=jnp.int32), jnp.array(curr_generation_idx, dtype=jnp.int32), token_types, rope_pos_ids, attn_bias)
        next_token_id = int(next_token_id)
        curr_generation_idx += 1
        running_text_ids[curr_generation_idx] = next_token_id
        if next_token_id == new_token_ids['eos_token_id']:
            eos_predicted = True

    generated_text_ids = running_text_ids[num_tokens_in_prompt+1 : curr_generation_idx] # skips bos and eos tokens
    generated_text = tokenizer.decode(generated_text_ids)

    return generated_text

###############################################################
######################## Text -> Image ########################
###############################################################
# #prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
# prompt = "A lantern glowing with green light, photorealistic"
# gen_img_latent = text2image(prompt, (1024, 1024))

# gen_img = ae_decode(ae_variables, gen_img_latent)
# gen_img = np.array(gen_img)[0]
# gen_img = np.clip((gen_img + 1) * 127.5, 0, 255).astype(np.uint8)
# gen_img = Image.fromarray(gen_img)

# gen_img.save("samples/a_green_lantern.png")

###############################################################
######################## Image Editing ########################
###############################################################
# source_image = Image.open("samples/woman.jpg")
# prompt = "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."
# gen_img_latent = image_editing(prompt, source_image)

# gen_img = ae_decode(ae_variables, gen_img_latent)
# gen_img = np.array(gen_img)[0]
# gen_img = np.clip((gen_img + 1) * 127.5, 0, 255).astype(np.uint8)
# gen_img = Image.fromarray(gen_img)

# gen_img.save("samples/woman_in_subway.png")

###############################################################
############################# VQA #############################
###############################################################
source_image = Image.open("samples/meme.jpg")
prompt = "Can someone explain what’s funny about this meme??"
model_response = vqa(prompt, source_image)
print(model_response)