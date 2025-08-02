#!/usr/bin/env python3
"""
Convert Flux auto-encoder weights (.safetensors) → Flax checkpoint.

Run:
    python ae_pt_to_flax_ckpt.py \
           --safetensors flux_ae.safetensors \
           --ckpt_dir    flux_ae_flax_ckpt
"""
from __future__ import annotations
import argparse, pathlib, re
from typing import Tuple, Dict

import torch
from safetensors.torch import load_file as load_sft

import jax, jax.numpy as jnp
from flax import traverse_util
from flax.core import FrozenDict
from flax.training import checkpoints

from flux_autoencoder_flax import build_autoencoder   # your Flax port

# ------------------------------------------------------------------
#  Hyper-params that define the block layout
# ------------------------------------------------------------------
NUM_RES_BLOCKS  = 2
CH_MULT         = [1, 2, 4, 4]
NUM_RESOLUTIONS = len(CH_MULT)
NUM_DEC_BLOCKS  = NUM_RES_BLOCKS + 1     # decoder: +1 block per up level

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
_leaf = dict(kernel="weight", scale="weight", bias="bias")

def _res(seg):   return dict(Conv_0="conv1", Conv_1="conv2", Conv_2="nin_shortcut",
                             GroupNorm_0="norm1", GroupNorm_1="norm2")[seg]
def _att(seg):   return dict(Conv_0="q", Conv_1="k", Conv_2="v", Conv_3="proj_out",
                             GroupNorm_0="norm")[seg]
def _top(seg, root):
    return dict(Conv_0=f"{root}.conv_in",
                Conv_1=f"{root}.conv_out",
                GroupNorm_0=f"{root}.norm_out")[seg]

def flax_to_pt(path: Tuple[str, ...]) -> str:
    root, second, *rest = path
    leaf = _leaf[rest[-1]]

    # ----- top-level conv / norm -----
    if second in ("Conv_0", "Conv_1", "GroupNorm_0"):
        return f"{_top(second, root)}.{leaf}"

    # ----- encoder ResNet blocks -----
    if root == "encoder" and second.startswith("ResnetBlock_"):
        idx = int(second.split("_")[1])
        total_down = NUM_RES_BLOCKS * NUM_RESOLUTIONS  # 8

        if idx >= total_down:                 # two mid blocks (idx 8,9)
            mid = idx - total_down + 1        # 1 or 2
            inside = _res(rest[0])
            return f"encoder.mid.block_{mid}.{inside}.{leaf}"

        level     = idx // NUM_RES_BLOCKS     # 0-3
        block_idx = idx %  NUM_RES_BLOCKS     # 0-1
        inside    = _res(rest[0])
        return f"encoder.down.{level}.block.{block_idx}.{inside}.{leaf}"

    # ----- encoder mid attention -----
    if root == "encoder" and second == "AttnBlock_0":
        inside = _att(rest[0])
        return f"encoder.mid.attn_1.{inside}.{leaf}"

    # ----- decoder mid -----
    if root == "decoder" and second in ("ResnetBlock_0", "ResnetBlock_1"):
        mid = int(second.split("_")[1]) + 1         # 1 or 2
        inside = _res(rest[0])
        return f"decoder.mid.block_{mid}.{inside}.{leaf}"
    if root == "decoder" and second == "AttnBlock_0":
        inside = _att(rest[0])
        return f"decoder.mid.attn_1.{inside}.{leaf}"

    # ----- decoder up blocks -----
    if root == "decoder" and second.startswith("ResnetBlock_"):
        idx   = int(second.split("_")[1]) - 2          # skip the 2 mid blocks
        level = idx // NUM_DEC_BLOCKS                  # 0..3  (Flax order)
        block_idx = idx % NUM_DEC_BLOCKS               # 0,1,2
        inside = _res(rest[0])

        pt_level = (NUM_RESOLUTIONS - 1) - level       # <<< flip  0→3, 1→2, 2→1, 3→0
        return f"decoder.up.{pt_level}.block.{block_idx}.{inside}.{leaf}"

    # ----- Down / Up sample convs -----
    if second.startswith("Downsample_"):
        lvl = int(second.split("_")[1])
        return f"{root}.down.{lvl}.downsample.conv.{leaf}"
    if second.startswith("Upsample_"):
        lvl = int(second.split("_")[1])            # Flax index (0,1,2)
        pt_lvl = (NUM_RESOLUTIONS - 1) - lvl       # 3→0, 2→1, 1→2, 0→3
        return f"{root}.up.{pt_lvl}.upsample.conv.{leaf}"

    raise ValueError("Unmapped path: " + "/".join(path))

def copy_params(pt: Dict[str, torch.Tensor], flax: FrozenDict) -> FrozenDict:
    flat_f = traverse_util.flatten_dict(flax, sep="/")
    out    = {}
    for k, arr in flat_f.items():
        pt_k = flax_to_pt(tuple(k.split("/")))
        if pt_k not in pt:
            raise KeyError(f"Missing tensor in PT state-dict: {pt_k}")
        t = pt[pt_k].cpu().numpy()
        if k.endswith("/kernel") and t.ndim == 4:
            t = t.transpose(2,3,1,0)          # (out,in,kh,kw) → (kh,kw,in,out)
        out[k] = jnp.asarray(t)
    return FrozenDict(traverse_util.unflatten_dict(
        {tuple(k.split("/")): v for k, v in out.items()}
    ))

# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True, help=".safetensors checkpoint")
    ap.add_argument("--ckpt_dir",    default="/home/pranav/bagel-jax/jax_port/flux_ae_flax_ckpt", help="Output dir")
    ap.add_argument("--step", type=int, default=0, help="Checkpoint step number")
    args = ap.parse_args()

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("✓ load PyTorch weights …")
    pt_state = load_sft(args.safetensors)

    print("✓ build Flax model …")
    ae      = build_autoencoder(sample_latent=True)
    params0 = ae.init(jax.random.PRNGKey(0), jnp.zeros((1,256,256,3)))["params"]

    print("✓ copy tensors …")
    params = copy_params(pt_state, params0)

    print("✓ save Flax checkpoint →", ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir, target=params, step=args.step, overwrite=True)
    print("Done ✔")

if __name__ == "__main__":
    main()
