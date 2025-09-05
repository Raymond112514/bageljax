import os
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict
from typing import Any, Callable, Optional, Tuple
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.core import freeze, unfreeze
import numpy as np
import optax
import torch
from safetensors.torch import load_file as load_sft
from copy import deepcopy

from bageljax.autoencoder import build_autoencoder

flax.config.update('flax_use_orbax_checkpointing', False)

ae = build_autoencoder(sample_latent=True)
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
ae_variables = ae.init(key, jnp.zeros((1, 1024, 1024, 3)))
print("Autoencoder initialized.")

# Print total number of parameters of the model
print("Total model parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(ae_variables["params"])]))

# Load the pytorch parameters
pytorch_param_path = "/home/pranav/bageljax/ae.safetensors"
pytorch_state_dict = load_sft(pytorch_param_path)
print("Loaded PyTorch parameters from", pytorch_param_path)

# -----------------------------
# Helpers to flatten JAX params
# -----------------------------
def _flatten_params(d, prefix=""):
    items = []
    for k, v in (d.items() if isinstance(d, (dict, FrozenDict)) else []):
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, (dict, FrozenDict)):
            items.extend(_flatten_params(v, name))
        else:
            items.append((name, v))
    return items

jax_params_list = _flatten_params(ae_variables["params"])
jax_params = {k: v for k, v in jax_params_list}
jax_keys_set = set(jax_params.keys())
pt_keys_set  = set(pytorch_state_dict.keys())

# -----------------------------------------
# Convenience: register and assert presence
# -----------------------------------------
pytorch_jax_parameter_mapping = []  # (pt_key, jax_key, is_conv_weight)

def add_map(pt_key: str, jax_key: str, is_conv_weight: bool):
    # existence checks now so we catch typos early
    assert pt_key in pt_keys_set, f"PT key missing: {pt_key}"
    assert jax_key in jax_keys_set, f"JAX key missing: {jax_key}"
    pytorch_jax_parameter_mapping.append((pt_key, jax_key, is_conv_weight))

def add_gn_pair(pt_prefix: str, jax_prefix: str):
    # GroupNorm: PT "weight"/"bias"  <->  Flax "scale"/"bias"
    add_map(f"{pt_prefix}.weight", f"{jax_prefix}/scale", False)
    add_map(f"{pt_prefix}.bias",   f"{jax_prefix}/bias",  False)

def add_conv_pair(pt_prefix: str, jax_prefix: str):
    # Conv2d: PT "weight"/"bias"  <->  Flax "kernel"/"bias"
    add_map(f"{pt_prefix}.weight", f"{jax_prefix}/kernel", True)
    add_map(f"{pt_prefix}.bias",   f"{jax_prefix}/bias",   False)  # bias ≠ transposed

def maybe_add_shortcut(pt_prefix: str, jax_prefix: str):
    # Optional 1x1 conv for channel change
    w_key = f"{pt_prefix}.nin_shortcut.weight"
    b_key = f"{pt_prefix}.nin_shortcut.bias"
    if w_key in pt_keys_set and f"{jax_prefix}/Conv_2/kernel" in jax_keys_set:
        add_map(w_key, f"{jax_prefix}/Conv_2/kernel", True)
        add_map(b_key, f"{jax_prefix}/Conv_2/bias",   False)

# -----------------
# ENCODER MAPPINGS
# -----------------
# stem
add_conv_pair("encoder.conv_in", "encoder/Conv_0")

# per-resolution ResNet blocks (4 levels, 2 blocks each)
# ResnetBlock indices in Flax: level 0 -> RB_0,1 ; level 1 -> RB_2,3 ; level 2 -> RB_4,5 ; level 3 -> RB_6,7
for lvl in range(4):
    for blk in range(2):
        rb_idx = 2 * lvl + blk
        pt_prefix  = f"encoder.down.{lvl}.block.{blk}"
        jax_prefix = f"encoder/ResnetBlock_{rb_idx}"
        add_gn_pair (f"{pt_prefix}.norm1", f"{jax_prefix}/GroupNorm_0")
        add_conv_pair(f"{pt_prefix}.conv1", f"{jax_prefix}/Conv_0")
        add_gn_pair (f"{pt_prefix}.norm2", f"{jax_prefix}/GroupNorm_1")
        add_conv_pair(f"{pt_prefix}.conv2", f"{jax_prefix}/Conv_1")
        maybe_add_shortcut(pt_prefix, jax_prefix)

# downsamplers at levels 0,1,2
for lvl in range(3):
    add_conv_pair(f"encoder.down.{lvl}.downsample.conv", f"encoder/Downsample_{lvl}/Conv_0")

# mid: RB -> Attn -> RB
add_gn_pair ( "encoder.mid.block_1.norm1", "encoder/ResnetBlock_8/GroupNorm_0")
add_conv_pair("encoder.mid.block_1.conv1", "encoder/ResnetBlock_8/Conv_0")
add_gn_pair ( "encoder.mid.block_1.norm2", "encoder/ResnetBlock_8/GroupNorm_1")
add_conv_pair("encoder.mid.block_1.conv2", "encoder/ResnetBlock_8/Conv_1")

# AttnBlock q,k,v,proj_out in creation order -> Conv_0..3
add_gn_pair ( "encoder.mid.attn_1.norm", "encoder/AttnBlock_0/GroupNorm_0")
add_conv_pair("encoder.mid.attn_1.q",   "encoder/AttnBlock_0/Conv_0")
add_conv_pair("encoder.mid.attn_1.k",   "encoder/AttnBlock_0/Conv_1")
add_conv_pair("encoder.mid.attn_1.v",   "encoder/AttnBlock_0/Conv_2")
add_conv_pair("encoder.mid.attn_1.proj_out", "encoder/AttnBlock_0/Conv_3")

add_gn_pair ( "encoder.mid.block_2.norm1", "encoder/ResnetBlock_9/GroupNorm_0")
add_conv_pair("encoder.mid.block_2.conv1", "encoder/ResnetBlock_9/Conv_0")
add_gn_pair ( "encoder.mid.block_2.norm2", "encoder/ResnetBlock_9/GroupNorm_1")
add_conv_pair("encoder.mid.block_2.conv2", "encoder/ResnetBlock_9/Conv_1")

# tail
add_gn_pair ( "encoder.norm_out", "encoder/GroupNorm_0")
add_conv_pair("encoder.conv_out", "encoder/Conv_1")

# ----------------
# DECODER MAPPINGS
# ----------------
# stem (z -> block_in)
add_conv_pair("decoder.conv_in", "decoder/Conv_0")

# mid: RB -> Attn -> RB
add_gn_pair ( "decoder.mid.block_1.norm1", "decoder/ResnetBlock_0/GroupNorm_0")
add_conv_pair("decoder.mid.block_1.conv1", "decoder/ResnetBlock_0/Conv_0")
add_gn_pair ( "decoder.mid.block_1.norm2", "decoder/ResnetBlock_0/GroupNorm_1")
add_conv_pair("decoder.mid.block_1.conv2", "decoder/ResnetBlock_0/Conv_1")

add_gn_pair ( "decoder.mid.attn_1.norm", "decoder/AttnBlock_0/GroupNorm_0")
add_conv_pair("decoder.mid.attn_1.q",   "decoder/AttnBlock_0/Conv_0")
add_conv_pair("decoder.mid.attn_1.k",   "decoder/AttnBlock_0/Conv_1")
add_conv_pair("decoder.mid.attn_1.v",   "decoder/AttnBlock_0/Conv_2")
add_conv_pair("decoder.mid.attn_1.proj_out", "decoder/AttnBlock_0/Conv_3")

add_gn_pair ( "decoder.mid.block_2.norm1", "decoder/ResnetBlock_1/GroupNorm_0")
add_conv_pair("decoder.mid.block_2.conv1", "decoder/ResnetBlock_1/Conv_0")
add_gn_pair ( "decoder.mid.block_2.norm2", "decoder/ResnetBlock_1/GroupNorm_1")
add_conv_pair("decoder.mid.block_2.conv2", "decoder/ResnetBlock_1/Conv_1")

# up path (levels 3,2,1,0) -> Flax ResnetBlock indices:
# level 3 -> RB_2,3,4  (+ Upsample_0)
# level 2 -> RB_5,6,7  (+ Upsample_1)
# level 1 -> RB_8,9,10 (+ Upsample_2)
# level 0 -> RB_11,12,13
for lvl in [3, 2, 1, 0]:
    base = 2 + (3 - lvl) * 3
    for blk in range(3):
        pt_prefix  = f"decoder.up.{lvl}.block.{blk}"
        jax_prefix = f"decoder/ResnetBlock_{base + blk}"
        add_gn_pair (f"{pt_prefix}.norm1", f"{jax_prefix}/GroupNorm_0")
        add_conv_pair(f"{pt_prefix}.conv1", f"{jax_prefix}/Conv_0")
        add_gn_pair (f"{pt_prefix}.norm2", f"{jax_prefix}/GroupNorm_1")
        add_conv_pair(f"{pt_prefix}.conv2", f"{jax_prefix}/Conv_1")
        maybe_add_shortcut(pt_prefix, jax_prefix)

    # upsamplers for levels 3,2,1
    if lvl != 0:
        up_idx = 3 - lvl  # 3->0, 2->1, 1->2
        add_conv_pair(f"decoder.up.{lvl}.upsample.conv", f"decoder/Upsample_{up_idx}/Conv_0")

# tail
add_gn_pair ( "decoder.norm_out", "decoder/GroupNorm_0")
add_conv_pair("decoder.conv_out", "decoder/Conv_1")

print(f"Mapping entries: {len(pytorch_jax_parameter_mapping)}")

# -----------------
# VALIDATION CHECKS
# -----------------
# 1) Duplicates (one-to-one)
pt_mapped_keys  = [pt for pt, _, _ in pytorch_jax_parameter_mapping]
jax_mapped_keys = [jx for _, jx, _ in pytorch_jax_parameter_mapping]

dupe_pt  = {k for k in pt_mapped_keys  if pt_mapped_keys.count(k)  > 1}
dupe_jax = {k for k in jax_mapped_keys if jax_mapped_keys.count(k) > 1}
assert not dupe_pt,  f"Duplicate PT keys in mapping: {sorted(dupe_pt)[:10]}"
assert not dupe_jax, f"Duplicate JAX keys in mapping: {sorted(dupe_jax)[:10]}"

# 2) Coverage (no leftovers in either side)
pt_unmapped = pt_keys_set  - set(pt_mapped_keys)
jax_unmapped = jax_keys_set - set(jax_mapped_keys)
assert not pt_unmapped,  f"Unmapped PT keys ({len(pt_unmapped)}):\n  " + "\n  ".join(sorted(pt_unmapped)[:50])
assert not jax_unmapped, f"Unmapped JAX keys ({len(jax_unmapped)}):\n  " + "\n  ".join(sorted(jax_unmapped)[:50])

# 3) Shape checks (with correct conv transpose)
def _pt_to_jax_shape(pt_arr, is_conv):
    s = tuple(pt_arr.shape)
    if is_conv:
        assert pt_arr.ndim == 4, f"Expected 4D conv weight, got {pt_arr.ndim}D for shape {s}"
        # PT: (OC, IC, KH, KW) -> Flax: (KH, KW, IC, OC)
        return (s[2], s[3], s[1], s[0])
    return s  # biases + GroupNorm params are identical

shape_mismatches = []
for pt_key, jx_key, is_conv in pytorch_jax_parameter_mapping:
    pt_arr = pytorch_state_dict[pt_key]
    jx_arr = jax_params[jx_key]
    exp_shape = _pt_to_jax_shape(pt_arr, is_conv)
    if tuple(jx_arr.shape) != tuple(exp_shape):
        shape_mismatches.append((pt_key, pt_arr.shape, jx_key, jx_arr.shape, is_conv))

assert not shape_mismatches, "Shape mismatches:\n" + "\n".join(
    [f"- {pt_key} {pt_shape} -> {jx_key} {jx_shape} (is_conv={is_conv})"
     for (pt_key, pt_shape, jx_key, jx_shape, is_conv) in shape_mismatches]
)

print("✅ Mapping looks consistent: one-to-one, full coverage, and shapes match (with conv transpose).")

# ------------------------
# Apply mapping → new params
# ------------------------
def _set_by_path(d, path, value):
    """Set leaf in nested dict by 'a/b/c' path."""
    keys = path.split("/")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value

def _pt_to_flax_array(pt_tensor, like_array, is_conv: bool):
    """Convert PT tensor → numpy and permute if conv; cast to target dtype."""
    x = pt_tensor.detach().cpu().numpy()
    if is_conv:
        # PT (OC, IC, KH, KW) -> Flax (KH, KW, IC, OC)
        x = np.transpose(x, (2, 3, 1, 0))
    # match dtype of existing flax param
    return x.astype(like_array.dtype, copy=False)

def build_flax_params_from_pt():
    # start from initialized params so structure matches exactly
    new_params = unfreeze(ae_variables["params"]).copy()

    # (Optional) sanity: ensure mapping covers all keys as asserted earlier.
    for pt_key, jx_key, is_conv in pytorch_jax_parameter_mapping:
        pt_t = pytorch_state_dict[pt_key]
        jx_like = jax_params[jx_key]
        arr = _pt_to_flax_array(pt_t, jx_like, is_conv)
        # additional shape guard (should already be validated)
        assert arr.shape == jx_like.shape, (
            f"Shape mismatch after convert: {pt_key}->{jx_key} "
            f"{arr.shape} vs {jx_like.shape}"
        )
        _set_by_path(new_params, jx_key, arr)

    return freeze(new_params)

# ------------------------
# Create TrainState w/ PT weights
# ------------------------
converted_params = build_flax_params_from_pt()

# If you already have a TrainState, you can replace its params.
# Otherwise, create one (optimizer choice doesn't matter for inference-only).
tx = optax.adam(1e-4)
train_state = TrainState.create(
    apply_fn=ae.apply,
    params=converted_params,
    tx=tx,
)

# ------------------------
# Save as a Flax checkpoint
# ------------------------
print("Saving Jax checkpoint...")
save_dir = "/home/pranav/bageljax/new_ae_ckpt"
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = checkpoints.save_checkpoint(
    save_dir,
    train_state,
    step=0,
    keep=10_000_000,  # keep must be an int
)
print("Saved Jax checkpoint to", checkpoint_path)

# ------------------------
# Final verify (optional)
# ------------------------
# Run a tiny forward to ensure params are loadable and numerically finite.
dummy_x = jnp.zeros((1, 256, 256, 3), dtype=jnp.float32)
_ = ae.apply({"params": train_state.params}, dummy_x, rngs={"gaussian": jax.random.PRNGKey(0)})
print("✅ Forward pass with converted params succeeded.")
