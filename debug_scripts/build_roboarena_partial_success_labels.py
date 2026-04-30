#!/usr/bin/env python3
"""Build label TFRecords with partial_success from RoboArena HF metadata.

For each example in input RoboArena shards, this script:
1) reads `rel_path` to identify session_id and policy side (A/B),
2) loads `evaluation_sessions/<session_id>/metadata.yaml` from HF repo,
3) extracts `policies.<A/B>.partial_success`,
4) writes a label-only TFRecord example with:
   - rewards: float list of length traj_len, filled with partial_success
   - partial_success: scalar float list [value]
   - rel_path: copied bytes for traceability

Output shards keep the same filename/order as input shards, so
example indices remain aligned with source TFRecords.
"""

import argparse
import os
from typing import Dict, Tuple

import tensorflow as tf
import yaml
from huggingface_hub import hf_hub_download


HF_REPO_ID = "RoboArena/DataDump_02-03-2026"
HF_REPO_TYPE = "dataset"


def print_green(text: str) -> None:
    print(f"\033[92m{text}\033[0m")


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _parse_rel_path(rel_path: str) -> Tuple[str, str]:
    # Example: evaluation_sessions/<session_id>/A_paligemma_fast_droid
    parts = rel_path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "evaluation_sessions":
        raise ValueError(f"Unexpected rel_path format: {rel_path}")
    session_id = parts[1]
    policy_segment = parts[2]
    policy_side = policy_segment.split("_", 1)[0]
    if len(policy_side) != 1 or not policy_side.isalpha() or not policy_side.isupper():
        raise ValueError(f"Could not infer policy side from rel_path: {rel_path}")
    return session_id, policy_side


def _read_partial_success(
    session_id: str,
    policy_side: str,
    revision: str,
    cache: Dict[str, dict],
) -> float:
    if session_id not in cache:
        metadata_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=f"evaluation_sessions/{session_id}/metadata.yaml",
            revision=revision,
        )
        with tf.io.gfile.GFile(metadata_path, "r") as f:
            cache[session_id] = yaml.safe_load(f)

    metadata = cache[session_id]
    policies = metadata.get("policies", {})
    if policy_side not in policies:
        raise KeyError(f"Policy {policy_side} missing in session {session_id}")
    value = float(policies[policy_side]["partial_success"])
    return value


def _parse_source_example(serialized: bytes) -> Tuple[str, int]:
    ex = tf.train.Example()
    ex.ParseFromString(serialized)
    feats = ex.features.feature

    rel_path_values = feats["rel_path"].bytes_list.value
    if not rel_path_values:
        raise ValueError("Missing rel_path bytes_list")
    rel_path = rel_path_values[0].decode("utf-8")

    traj_len_values = feats.get("traj_len", tf.train.Feature()).int64_list.value
    if traj_len_values:
        traj_len = int(traj_len_values[0])
    else:
        # Fallback: try number of shoulder images if traj_len absent.
        traj_len = len(feats["shoulder_image_1"].bytes_list.value)
    if traj_len <= 0:
        raise ValueError(f"Invalid traj_len={traj_len} for rel_path={rel_path}")
    return rel_path, traj_len


def _build_label_example(rel_path: str, traj_len: int, partial_success: float, example_index: int):
    rewards = [partial_success] * traj_len
    out = tf.train.Example(
        features=tf.train.Features(
            feature={
                "rewards": _float_feature(rewards),
                "partial_success": _float_feature([partial_success]),
                "traj_len": _int64_feature([traj_len]),
                "rel_path": _bytes_feature([rel_path.encode("utf-8")]),
                "example_index": _int64_feature([example_index]),
            }
        )
    )
    return out


def _normalize_gcs_prefix(prefix: str) -> str:
    if prefix.startswith("gs://") or "://" in prefix:
        return prefix.rstrip("/")
    return f"gs://{prefix.rstrip('/')}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create RoboArena label TFRecords using HF metadata partial_success.",
    )
    parser.add_argument(
        "--input_prefix",
        type=str,
        default="gs://raymond-us-west1/droid/roboarena",
        help="Input TFRecord prefix (raw RoboArena shards).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="gs://raymond-us-west1/droid_labeled/roboarena_partial_success",
        help="Output TFRecord prefix for label-only shards.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="roboarena-*.tfrecord",
        help="Glob under input_prefix to select shards.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default="main",
        help="HF dataset revision/branch/tag.",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="If >0, process only first N shards (debug mode).",
    )
    args = parser.parse_args()

    in_prefix = _normalize_gcs_prefix(args.input_prefix)
    out_prefix = _normalize_gcs_prefix(args.output_prefix)
    input_pattern = f"{in_prefix}/{args.glob}"
    shard_paths = sorted(tf.io.gfile.glob(input_pattern))
    if not shard_paths:
        raise ValueError(f"No input shards matched: {input_pattern}")
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    tf.io.gfile.makedirs(out_prefix)
    metadata_cache: Dict[str, dict] = {}
    total_examples = 0

    print(f"num_shards: {len(shard_paths)}")
    print(f"input_pattern: {input_pattern}")
    print(f"output_prefix: {out_prefix}")

    for shard_i, in_path in enumerate(shard_paths):
        shard_name = os.path.basename(in_path)
        out_path = f"{out_prefix}/{shard_name}"
        count_this_shard = 0
        with tf.io.TFRecordWriter(out_path) as writer:
            for example_index, raw in enumerate(tf.data.TFRecordDataset([in_path])):
                rel_path, traj_len = _parse_source_example(raw.numpy())
                session_id, policy_side = _parse_rel_path(rel_path)
                print(f"session_id: {session_id}")
                partial_success = _read_partial_success(
                    session_id=session_id,
                    policy_side=policy_side,
                    revision=args.hf_revision,
                    cache=metadata_cache,
                )
                labeled = _build_label_example(
                    rel_path=rel_path,
                    traj_len=traj_len,
                    partial_success=partial_success,
                    example_index=example_index,
                )
                writer.write(labeled.SerializeToString())
                count_this_shard += 1
        total_examples += count_this_shard
        print_green(
            f"[{shard_i + 1}/{len(shard_paths)}] done {shard_name}: {count_this_shard} trajectories"
        )

    print(f"total_trajectories_written: {total_examples}")
    print(f"unique_sessions_cached: {len(metadata_cache)}")


if __name__ == "__main__":
    main()
