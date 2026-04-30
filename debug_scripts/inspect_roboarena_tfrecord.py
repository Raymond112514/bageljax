#!/usr/bin/env python3
"""Inspect one trajectory from a RoboArena TFRecord shard.

Example:
  python debug_scripts/inspect_roboarena_tfrecord.py \
    --tfrecord_path gs://raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord
"""

import argparse
from typing import Iterable

import tensorflow as tf


def _normalize_tfrecord_path(path: str) -> str:
    if path.startswith("gs://") or path.startswith("/") or "://" in path:
        return path
    return f"gs://{path}"


def _safe_utf8_preview(blob: bytes, max_chars: int = 120) -> str:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _format_numeric_preview(values: Iterable, preview_count: int) -> str:
    values = list(values)
    if not values:
        return "[]"
    shown = values[:preview_count]
    suffix = " ..." if len(values) > preview_count else ""
    return f"{shown}{suffix}"


def _print_feature(name: str, feature: tf.train.Feature, preview_count: int) -> None:
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        items = list(feature.bytes_list.value)
        print(f"- {name}: bytes_list (count={len(items)})")
        for i, blob in enumerate(items[:preview_count]):
            utf8_preview = _safe_utf8_preview(blob)
            line = f"  [{i}] len={len(blob)}"
            if utf8_preview:
                line += f" utf8='{utf8_preview}'"
            print(line)
        if len(items) > preview_count:
            print(f"  ... {len(items) - preview_count} more entries")
        return

    if kind == "float_list":
        values = list(feature.float_list.value)
        print(
            f"- {name}: float_list (count={len(values)}) "
            f"preview={_format_numeric_preview(values, preview_count)}"
        )
        return

    if kind == "int64_list":
        values = list(feature.int64_list.value)
        print(
            f"- {name}: int64_list (count={len(values)}) "
            f"preview={_format_numeric_preview(values, preview_count)}"
        )
        return

    print(f"- {name}: <unknown feature kind>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read one trajectory from a TFRecord and print all features.",
    )
    parser.add_argument(
        "--tfrecord_path",
        type=str,
        default="raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord",
        help="Local or GCS TFRecord path. If no scheme is provided, gs:// is assumed.",
    )
    parser.add_argument(
        "--example_index",
        type=int,
        default=0,
        help="0-based index of trajectory/example to inspect within the shard.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=3,
        help="How many values (or bytes entries) to preview for each feature.",
    )
    args = parser.parse_args()

    tfrecord_path = _normalize_tfrecord_path(args.tfrecord_path)
    print(f"Reading TFRecord: {tfrecord_path}")
    print(f"Target example index: {args.example_index}")

    ds = tf.data.TFRecordDataset([tfrecord_path])
    raw_example = next(iter(ds.skip(args.example_index).take(1)), None)
    if raw_example is None:
        raise ValueError(
            f"No example found at index {args.example_index} in {tfrecord_path}."
        )

    example = tf.train.Example()
    example.ParseFromString(raw_example.numpy())
    features = example.features.feature

    print(f"Found {len(features)} features:")
    for name in sorted(features.keys()):
        _print_feature(name, features[name], args.preview_count)


if __name__ == "__main__":
    main()
