# pytree_report.py
# Tiny helpers to pretty-print and diff JAX parameter pytrees without materializing arrays.

from __future__ import annotations
import io
import os
from collections import defaultdict
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu

# ---- path string helpers -----------------------------------------------------

try:
    # Newer JAX exposes these key types
    from jax.tree_util import DictKey, SequenceKey, GetAttrKey, FlattenedIndexKey
except Exception:
    DictKey = SequenceKey = GetAttrKey = FlattenedIndexKey = object  # fallback

def _path_to_str(path) -> str:
    """Convert a JAX KeyPath (tuple of Key objects) to a stable 'a/b/0/c' style string."""
    parts = []
    for k in path:
        if isinstance(k, DictKey):
            parts.append(str(k.key))
        elif isinstance(k, SequenceKey):
            parts.append(str(k.idx))
        elif isinstance(k, GetAttrKey):
            parts.append(k.name)
        elif isinstance(k, FlattenedIndexKey):
            parts.append(str(k.key))
        else:
            # Fallback: last resort stringification
            parts.append(str(k))
    return "/".join(parts) if parts else "<root>"

# ---- formatting helpers ------------------------------------------------------

def _fmt_shape(x) -> str:
    try:
        return "(" + ",".join(str(d) for d in x.shape) + ")"
    except Exception:
        return "-"

def _fmt_bytes(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0

def _is_array(x) -> bool:
    # Works across jax.Array, DeviceArray, and numpy arrays without importing private types.
    return hasattr(x, "shape") and hasattr(x, "dtype")

def _safe_nbytes(x) -> int:
    try:
        # Handle jnp.bfloat16, etc.
        itemsize = np.dtype(x.dtype).itemsize
        return int(itemsize * (int(np.prod(x.shape)) if x.shape else 1))
    except Exception:
        return 0

def _sharding_str(x) -> str:
    try:
        s = getattr(x, "sharding", None)
        if s is None:
            return "-"
        # Try to print a concise sharding summary
        spec = getattr(s, "spec", None)
        if spec is not None:
            return f"{type(s).__name__}:{spec}"
        return type(s).__name__
    except Exception:
        return "-"

# ---- core collectors ---------------------------------------------------------

def _flatten_with_metadata(tree):
    """Yield tuples: (path_str, leaf, meta_dict)."""
    entries = []
    for path, leaf in jtu.tree_flatten_with_path(tree)[0]:
        p = _path_to_str(path)
        if _is_array(leaf):
            shape = tuple(getattr(leaf, "shape", ()))
            dtype = getattr(leaf, "dtype", None)
            nelems = int(np.prod(shape)) if shape else 1
            nbytes = _safe_nbytes(leaf)
            shard = _sharding_str(leaf)
            entries.append((
                p,
                leaf,
                dict(kind="array", shape=shape, dtype=dtype, nelems=nelems, nbytes=nbytes, sharding=shard),
            ))
        else:
            entries.append((p, leaf, dict(kind="other", repr=repr(leaf))))
    # Sort by path for stable output
    entries.sort(key=lambda t: t[0])
    return entries

# ---- public API: write report ------------------------------------------------

def write_pytree_report(tree, out_path: str, *, title: str | None = None, include_sharding: bool = True) -> None:
    """
    Write a neatly formatted report of a JAX pytree to a text file.

    Columns for arrays: PATH | SHAPE | DTYPE | #ELEMS | BYTES | [SHARDING?]
    Non-array leaves are listed with a short repr.

    This function never materializes device arrays to host; it only reads metadata.
    """
    entries = _flatten_with_metadata(tree)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Column widths
    path_w = max(8, min(140, max((len(p) for p, _, _ in entries), default=8)))
    shape_w = 22
    dtype_w = 10
    elems_w = 14
    bytes_w = 12

    # Totals
    total_params = 0
    total_bytes = 0
    dtype_totals = defaultdict(lambda: [0, 0])  # dtype -> [elems, bytes]

    buf = io.StringIO()
    hdr = title or "JAX Pytree Report"
    buf.write(f"{hdr}\n")
    buf.write(f"Generated: {now}\n")
    buf.write(f"Leaves: {len(entries)}\n\n")

    # Header row
    cols = [
        f"{'PATH':<{path_w}}",
        f"{'SHAPE':>{shape_w}}",
        f"{'DTYPE':>{dtype_w}}",
        f"{'#ELEMS':>{elems_w}}",
        f"{'BYTES':>{bytes_w}}",
    ]
    if include_sharding:
        cols.append("SHARDING")
    buf.write("  ".join(cols) + "\n")
    buf.write("-" * (path_w + shape_w + dtype_w + elems_w + bytes_w + (2 if include_sharding else 1) * 2 + 12) + "\n")

    # Rows
    for path, leaf, meta in entries:
        if meta["kind"] == "array":
            shape_s = _fmt_shape(leaf)
            dtype_s = str(meta["dtype"])
            nelems = meta["nelems"]
            nbytes = meta["nbytes"]
            total_params += nelems
            total_bytes += nbytes
            dtype_totals[dtype_s][0] += nelems
            dtype_totals[dtype_s][1] += nbytes

            row = [
                f"{path:<{path_w}}",
                f"{shape_s:>{shape_w}}",
                f"{dtype_s:>{dtype_w}}",
                f"{nelems:>{elems_w},}",
                f"{_fmt_bytes(nbytes):>{bytes_w}}",
            ]
            if include_sharding:
                row.append(meta["sharding"])
            buf.write("  ".join(row) + "\n")
        else:
            # Non-array leaf
            row = [
                f"{path:<{path_w}}",
                f"{'-':>{shape_w}}",
                f"{'-':>{dtype_w}}",
                f"{'-':>{elems_w}}",
                f"{'-':>{bytes_w}}",
            ]
            if include_sharding:
                row.append(meta["repr"])
            buf.write("  ".join(row) + "\n")

    # Summary
    buf.write("\nTotals (arrays only):\n")
    buf.write(f"  Parameters: {total_params:,}\n")
    buf.write(f"  Memory    : {_fmt_bytes(total_bytes)}\n")
    if dtype_totals:
        buf.write("  By dtype:\n")
        for dt, (elems, bts) in sorted(dtype_totals.items(), key=lambda kv: (-kv[1][1], kv[0])):
            buf.write(f"    - {dt:<10}  elems={elems:,}  bytes={_fmt_bytes(bts)}\n")

    # Write to disk
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

# ---- public API: diff two pytrees -------------------------------------------

def write_pytree_diff(tree_a, tree_b, out_path: str, *, name_a="A", name_b="B") -> None:
    """
    Compare two pytrees (by leaf paths) and write a diff report:
      - Leaves only in A
      - Leaves only in B
      - Leaves in both but with mismatched SHAPE or DTYPE
    """
    def _as_index(tree):
        idx = {}
        for path, leaf, meta in _flatten_with_metadata(tree):
            if meta["kind"] == "array":
                idx[path] = ("array", meta["shape"], str(meta["dtype"]))
            else:
                idx[path] = ("other", None, None)
        return idx

    idx_a = _as_index(tree_a)
    idx_b = _as_index(tree_b)

    only_a = sorted(set(idx_a) - set(idx_b))
    only_b = sorted(set(idx_b) - set(idx_a))
    common = sorted(set(idx_a) & set(idx_b))

    mismatches = []
    for p in common:
        kind_a, shape_a, dtype_a = idx_a[p]
        kind_b, shape_b, dtype_b = idx_b[p]
        if kind_a != kind_b:
            mismatches.append((p, f"kind: {kind_a} vs {kind_b}"))
        elif kind_a == "array":
            shape_diff = shape_a != shape_b
            dtype_diff = dtype_a != dtype_b
            if shape_diff or dtype_diff:
                msg = []
                if shape_diff: msg.append(f"shape: {shape_a} vs {shape_b}")
                if dtype_diff: msg.append(f"dtype: {dtype_a} vs {dtype_b}")
                mismatches.append((p, "; ".join(msg)))

    # write
    buf = io.StringIO()
    buf.write(f"Pytree Diff: {name_a} vs {name_b}\n")
    buf.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _section(title, items):
        buf.write(f"{title} ({len(items)}):\n")
        if not items:
            buf.write("  (none)\n\n")
            return
        for it in items:
            if isinstance(it, tuple):
                p, msg = it
                buf.write(f"  - {p}: {msg}\n")
            else:
                buf.write(f"  - {it}\n")
        buf.write("\n")

    _section(f"Only in {name_a}", only_a)
    _section(f"Only in {name_b}", only_b)
    _section("Mismatched in both", mismatches)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

# ---- examples ---------------------------------------------------------------

if __name__ == "__main__":
    # Fake trees for demonstration
    tree1 = {"params": {"Dense_0": {"kernel": jnp.zeros((32, 64), jnp.bfloat16),
                                    "bias": jnp.zeros((64,), jnp.float32)}}}
    tree2 = {"params": {"Dense_0": {"kernel": jnp.zeros((32, 128), jnp.bfloat16),  # shape changed
                                    "bias": jnp.zeros((64,), jnp.float32)},
                        "Dense_1": {"kernel": jnp.zeros((64, 10), jnp.float32)}}}  # new

    write_pytree_report(tree1, "tree1_report.txt", title="Model A")
    write_pytree_report(tree2, "tree2_report.txt", title="Model B")
    write_pytree_diff(tree1, tree2, "tree_diff.txt", name_a="Model A", name_b="Model B")
