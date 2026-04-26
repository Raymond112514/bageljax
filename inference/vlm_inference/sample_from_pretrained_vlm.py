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
from bageljax.mixture_of_transformers import MixtureOfTransformers
from bageljax.common import ModuleDict
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
    "mixture_of_transformers": MixtureOfTransformers(),
    "logits_head": LogitsHead(),
}

model_def = ModuleDict(networks)

def init_fn(rng):
    rng, init_rng = jax.random.split(rng)

    # For init, let's pick reasonable values of some of the input parameters
    B = 1 # batch size
    H, W = 672, 672 # image height and width, 672 is divisible by 14 and 16 ---> no longer needs to be divisible by 16
    llm_hidden_dim = 3584 # LLM hidden dimension
    L = 42*42 # sequence length, for llm2vae needs to be (H/16)*(W/16)

    params = model_def.init({'params': init_rng},
                                token_embedder = [
                                    jnp.zeros((B, L), dtype=jnp.int32),
                                ],
                                vision_encoder = [
                                    jnp.zeros((B, H, W, 3), dtype=jnp.bfloat16),
                                ],
                                mixture_of_transformers = [
                                    jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
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
# Tokenizer
################################################
tokenizer_load_path = "pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

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
    def greedily_predict_next_token(train_state, pre_llm_vit_tokens, text_ids, curr_generation_idx, rope_pos_ids, attn_bias):
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
        # Prepare token_types, used later for constructing the attention bias
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
        next_token_id = greedily_predict_next_token(train_state, pre_llm_vit_tokens, jnp.array(running_text_ids, dtype=jnp.int32), jnp.array(curr_generation_idx, dtype=jnp.int32), rope_pos_ids, attn_bias)
        next_token_id = int(next_token_id)
        curr_generation_idx += 1
        running_text_ids[curr_generation_idx] = next_token_id
        if next_token_id == new_token_ids['eos_token_id']:
            eos_predicted = True

    generated_text_ids = running_text_ids[num_tokens_in_prompt+1 : curr_generation_idx] # skips bos and eos tokens
    generated_text = tokenizer.decode(generated_text_ids)

    return generated_text

###############################################################
############################# VQA #############################
###############################################################
source_image = Image.open("samples/meme.jpg")
prompt = "Can someone explain what's funny about this meme??"
model_response = vqa(prompt, source_image)
print(model_response)