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

from bageljax.vocabulary import TokenEmbedder, LogitsHead
from bageljax.vision_encoder import VisionEncoder
from bageljax.vae2llm2vae import TimeEmbedder, VAE2LLM, LLM2VAE
from bageljax.mixture_of_transformers import MixtureOfTransformers
from bageljax.common import ModuleDict

# We will be saving checkpoints in this script, and we want to avoid using orbax
# Prevent flax.checkpoints from using Orbax backend
flax.config.update('flax_use_orbax_checkpointing', False)

# Create the model
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

rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
train_state = jax.jit(init_fn)(key)

# Print total number of parameters of the model
print("Total model parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(train_state.params)]))

# Print out the parameter pytree
def print_params(params, prefix=''):
    if isinstance(params, dict):
        for k, v in params.items():
            print_params(v, prefix + '/' + k if prefix else k)
    else:
        print(f"{prefix}: {params.shape}, dtype={params.dtype}")

# Let's print this out to a file
with open("model_params.txt", "w") as f:
    import sys
    original_stdout = sys.stdout
    sys.stdout = f
    print_params(train_state.params)
    sys.stdout = original_stdout

# Load the pytorch parameters
pytorch_param_path = "/raid/users/pranav/bagel_pytorch_checkpoint/BAGEL-7B-MoT/ema.safetensors"
pytorch_state_dict = load_sft(pytorch_param_path)
print("Loaded PyTorch parameters from", pytorch_param_path)

# Let's print out the pytorch param pytree in the same format as we've printed out for the Jax params
# We will use this to set up the mapping next

with open("pytorch_params.txt", "w") as f:
    import sys
    original_stdout = sys.stdout
    sys.stdout = f
    print_params(pytorch_state_dict)
    sys.stdout = original_stdout

pytorch_jax_parameter_mapping = [] # will be a list of tuples: (pytorch_key, jax_key, bool_transpose)

# Connector
pytorch_jax_parameter_mapping.append( ("connector.fc1.bias",
                                        "modules_vision_encoder/connector/connector_fc1/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("connector.fc1.weight",
                                        "modules_vision_encoder/connector/connector_fc1/kernel",
                                        True) )
pytorch_jax_parameter_mapping.append( ("connector.fc2.bias",
                                        "modules_vision_encoder/connector/connector_fc2/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("connector.fc2.weight",
                                        "modules_vision_encoder/connector/connector_fc2/kernel",
                                        True) )

# Token Embedder
pytorch_jax_parameter_mapping.append( ("language_model.model.embed_tokens.weight",
                                        "modules_token_embedder/weight",
                                        False) )

# Output head
pytorch_jax_parameter_mapping.append( ("language_model.lm_head.weight",
                                        "modules_logits_head/weight",
                                        False) )

# Mixture of Transformers (28 layers total)
for layer_idx in range(28):
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.input_layernorm.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/txt/input_rms/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.input_layernorm_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/gen/input_rms/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp.down_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/txt/mlp/txt/down_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp.gate_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/txt/mlp/txt/gate_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp.up_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/txt/mlp/txt/up_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp_moe_gen.down_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/gen/mlp/gen/down_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp_moe_gen.gate_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/gen/mlp/gen/gate_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.mlp_moe_gen.up_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/gen/mlp/gen/up_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/txt/post_attn_rms/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.post_attention_layernorm_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/gen/post_attn_rms/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_norm.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/k_norm/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_norm_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/k_norm/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_proj.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/k_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/k_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_proj_moe_gen.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/k_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.k_proj_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/k_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/o_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.o_proj_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/o_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_norm.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/q_norm/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_norm_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/q_norm/weight",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_proj.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/q_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/q_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_proj_moe_gen.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/q_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.q_proj_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/q_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.v_proj.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/v_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.v_proj.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/txt/v_proj/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.v_proj_moe_gen.bias",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/v_proj/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"language_model.model.layers.{layer_idx}.self_attn.v_proj_moe_gen.weight",
                                            f"modules_mixture_of_transformers/layer_{layer_idx}/attn/gen/v_proj/kernel",
                                            True) )

# Post MoT Norm
pytorch_jax_parameter_mapping.append( ("language_model.model.norm.weight",
                                        "modules_mixture_of_transformers/txt/final_rms/weight",
                                        False) )
pytorch_jax_parameter_mapping.append( ("language_model.model.norm_moe_gen.weight",
                                        "modules_mixture_of_transformers/gen/final_rms/weight",
                                        False) )

# VAE Position Embedding Table
pytorch_jax_parameter_mapping.append( ("latent_pos_embed.pos_embed",
                                        "modules_vae2llm/pos_embed",
                                        False) )

# LLM2VAE
pytorch_jax_parameter_mapping.append( ("llm2vae.bias",
                                        "modules_llm2vae/llm2vae/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("llm2vae.weight",
                                        "modules_llm2vae/llm2vae/kernel",
                                        True) )

# Time Embedder
pytorch_jax_parameter_mapping.append( ("time_embedder.mlp.0.bias",
                                        "modules_time_embedder/mlp/dense0/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("time_embedder.mlp.0.weight",
                                        "modules_time_embedder/mlp/dense0/kernel",
                                        True) )
pytorch_jax_parameter_mapping.append( ("time_embedder.mlp.2.bias",
                                        "modules_time_embedder/mlp/dense1/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("time_embedder.mlp.2.weight",
                                        "modules_time_embedder/mlp/dense1/kernel",
                                        True) )

# VAE2LLM
pytorch_jax_parameter_mapping.append( ("vae2llm.bias",
                                        "modules_vae2llm/vae2llm/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("vae2llm.weight",
                                        "modules_vae2llm/vae2llm/kernel",
                                        True) )

# ViT LLM Position Embedding Table
pytorch_jax_parameter_mapping.append( ("vit_pos_embed.pos_embed",
                                        "modules_vision_encoder/connector/post_vit_pos_embed",
                                        False) )

# ViT Patch Embeddings
pytorch_jax_parameter_mapping.append( ("vit_model.vision_model.embeddings.patch_embedding.bias",
                                        "modules_vision_encoder/vit/patch_embed/proj/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("vit_model.vision_model.embeddings.patch_embedding.weight",
                                        "modules_vision_encoder/vit/patch_embed/proj/kernel",
                                        True) )

# ViT Learned Positional Embeddings
pytorch_jax_parameter_mapping.append( ("vit_model.vision_model.embeddings.position_embedding.weight",
                                        "modules_vision_encoder/vit/pre_vit_pos_embed",
                                        False) )

# Post ViT LayerNorm
pytorch_jax_parameter_mapping.append( ("vit_model.vision_model.post_layernorm.bias",
                                        "modules_vision_encoder/vit/post_vit_ln/bias",
                                        False) )
pytorch_jax_parameter_mapping.append( ("vit_model.vision_model.post_layernorm.weight",
                                        "modules_vision_encoder/vit/post_vit_ln/scale",
                                        False) )

# ViT Transformer Layers (26 layers total)
for layer_idx in range(26):
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.layer_norm1.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/ln1/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.layer_norm1.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/ln1/scale",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.layer_norm2.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/ln2/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.layer_norm2.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/ln2/scale",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mlp/fc1/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mlp/fc1/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mlp/fc2/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mlp/fc2/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/k/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/k/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/out/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/out/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/q/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/q/kernel",
                                            True) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/v/bias",
                                            False) )
    pytorch_jax_parameter_mapping.append( (f"vit_model.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight",
                                            f"modules_vision_encoder/vit/block_{layer_idx}/mha/v/kernel",
                                            True) )

# Ok, now that we've defined the mapping, we need to go key by key through 
# the Jax model params (it's easier if we go through the jax first, since the pytorch 
# state dict is flat, while the jax params are a pytree)

# Create a reverse mapping from jax_key to (pytorch_key, bool_transpose)
jax_to_pytorch_map = {}
for pytorch_key, jax_key, bool_transpose in pytorch_jax_parameter_mapping:
    jax_to_pytorch_map[jax_key] = (pytorch_key, bool_transpose)

# Let's create a copy of the train_state params to modify
new_params = deepcopy(train_state.params)

def replace_params(params, prefix=''):
    if isinstance(params, dict):
        for k, v in params.items():
            replace_params(v, prefix + '//' + k if prefix else k) # note we use '//' because some keys have '/' in them
    else:
        # params is a leaf array
        # first let's find the corresponding pytorch key
        jax_key = prefix.replace('//', '/') # convert back to single '/', we created the mapping before we realized the '/' issue
        assert jax_key in jax_to_pytorch_map, f"Jax key {jax_key} not found in mapping!"
        pytorch_key, bool_transpose = jax_to_pytorch_map[jax_key]
        assert pytorch_key in pytorch_state_dict, f"PyTorch key {pytorch_key} not found in state dict!"
        pytorch_param = pytorch_state_dict[pytorch_key].to(torch.float32).numpy()
        if bool_transpose:
            pytorch_param = pytorch_param.T
        # Now check shapes match
        if params.shape != pytorch_param.shape:
            raise ValueError(f"Shape mismatch for {jax_key}: Jax shape {params.shape}, PyTorch shape {pytorch_param.shape}")
        # Convert to jax array
        pytorch_param_jax = jnp.array(pytorch_param, dtype=jnp.bfloat16)
        # Now replace in new_params
        # We need to navigate to the right place in new_params
        def set_in_dict(d, keys, value):
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = value
        keys = prefix.split('//')
        set_in_dict(new_params, keys, pytorch_param_jax)

replace_params(train_state.params)

# Now new_params should have all the parameters replaced
# Let's replace the train_state params
train_state = train_state.replace(params=new_params)

# Let's save the train_state as a flax checkpoint
print("Saving Jax checkpoint...")
save_dir = "/home/pranav/bageljax/pretrained_weights/bagel"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_path = checkpoints.save_checkpoint(
    save_dir, train_state, step=0, keep=1e7,
)
print("Saved Jax checkpoint to", checkpoint_path)