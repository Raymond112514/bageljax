from __future__ import annotations
from dataclasses import dataclass
import re
import random
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import tensorflow as tf

from bageljax.data.data_utils import (
    ACTION_NORM_EPS,
    binarize_gripper_action,
    default_joint_velocity_norm_tensors,
    normalize_joint_velocity_7d,
)

_TFRECORD_SHARD_RE = re.compile(r"-(\d+)\.tfrecord(?:\..*)?$")

@dataclass(frozen=True)
class TrajectoryObservation:
    """
    Observation for a single trajectory.
      - wrist_image: (T, 288, 512, 3)
      - shoulder_image: (T, 288, 512, 3)
      - language_instruction: (T,)
      - action/joint_velocity: (T, 8) — 7D joint velocity + binarized gripper (matches training)
      - image: (T, 672, 560, 3), shoulder-only after resize (matches training Dataset)
      - distance: (T,)
    """
    wrist_image: np.ndarray
    shoulder_image: np.ndarray
    language_instruction: np.ndarray
    action_joint_8d: np.ndarray
    image: np.ndarray
    distance: np.ndarray
    tfrecord_path: str
    tfrecord_file_index: int
    tfrecord_example_index: int
    tfrecord_shard_id: Optional[int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wrist_image": self.wrist_image,
            "shoulder_image": self.shoulder_image,
            "language_instruction": self.language_instruction,
            "action/joint_velocity": self.action_joint_8d,
            "image": self.image,
            "distance": self.distance,
            "tfrecord_path": self.tfrecord_path,
            "tfrecord_file_index": self.tfrecord_file_index,
            "tfrecord_example_index": self.tfrecord_example_index,
            "tfrecord_shard_id": self.tfrecord_shard_id,
        }

class TrajectoryDataSampler:
    """A minimal TFRecord trajectory sampler compatible with Droid and Roboarena dataset."""

    IMG_KEYS = (
        "wrist_image",
        "shoulder_image_1",
        "shoulder_image_2",
    )
    STR_KEYS = (
        "language_instruction1",
        "language_instruction2",
        "language_instruction3",
    )
    F32_KEYS = (
        "action/joint_velocity",
        "action/gripper_position",
    )

    def __init__(
        self,
        data_paths: Sequence[str],
        *,
        seed: int = 0,
        shoulder_stream: str = "shoulder_image_1",
        language_stream: str = "language_instruction1",
        image_resolution: tuple[int, int] = (672, 560),
    ):
        """
        Args:
          - data_paths: list of TFRecord file paths or glob patterns.
          - seed: RNG seed for picking which TFRecord file to sample from.
          - shoulder_stream: which shoulder key to use.
          - language_stream: which language key to use.
          - image_resolution: (H, W) after stacking + resize
        """
        if shoulder_stream not in self.IMG_KEYS:
            raise ValueError(f"Unsupported shoulder_stream={shoulder_stream}")
        if language_stream not in self.STR_KEYS:
            raise ValueError(f"Unsupported language_stream={language_stream}")
        self._jv_mean, self._jv_std = default_joint_velocity_norm_tensors()
        self._jv_eps = tf.constant(ACTION_NORM_EPS, dtype=tf.float32)
        self._rng = random.Random(int(seed))
        self._shoulder_stream = shoulder_stream
        self._language_stream = language_stream
        self._image_resolution = tuple(map(int, image_resolution))

        # Expand globs once; sampling then just picks one TFRecord file.
        paths: List[str] = []
        for p in data_paths:
            matches = tf.io.gfile.glob(p)
            if not matches:
                raise ValueError(f"No files match data path/glob: {p}")
            paths.extend(matches)
        paths = sorted(set(paths))
        if not paths:
            raise ValueError("No TFRecord files found in data_paths.")
        self._tfrecord_paths = paths

    def sample_one_trajectory(
        self,
        *,
        tfrecord_file_index: Optional[int] = None,
        example_index: int = 0,
        min_traj_len: int = 20,
        max_attempts: int = 50,
    ) -> TrajectoryObservation:
        """
        Sample one TFRecord example (one trajectory).
        Returns a TrajectoryObservation object.

        Args:
          - tfrecord_file_index: if None, choose a random TFRecord file.
          - example_index: index of the example within the selected TFRecord file.
          - min_traj_len: minimum trajectory length.
          - max_attempts: how many consecutive examples to try before giving up.
        """
        if tfrecord_file_index is None:
            tfrecord_file_index = self._rng.randrange(len(self._tfrecord_paths))
        tfrecord_file_index = int(tfrecord_file_index)
        tfrecord_path = self._tfrecord_paths[tfrecord_file_index]
        m = _TFRECORD_SHARD_RE.search(tfrecord_path)
        tfrecord_shard_id = int(m.group(1)) if m is not None else None

        # Try consecutive examples until we find one that passes the length filter.
        current_example_index = int(example_index)
        last_wrist_len: Optional[int] = None

        for _ in range(int(max_attempts)):
            ds = tf.data.TFRecordDataset([tfrecord_path]).skip(current_example_index).take(1)
            try:
                serialized_example = next(iter(ds))
            except StopIteration:
                break

            traj = self._decode_trajectory(serialized_example)

            wrist_T = int(tf.shape(traj["wrist_image"])[0].numpy())
            last_wrist_len = wrist_T
            if wrist_T >= int(min_traj_len):
                shoulder = traj[self._shoulder_stream]
                wrist = traj["wrist_image"]
                instruction = traj[self._language_stream]  # [T] bytes

                # Shoulder only
                shoulder = tf.ensure_shape(shoulder, [None, 288, 512, 3])
                image = tf.cast(
                    tf.round(
                        tf.image.resize(
                            shoulder,
                            self._image_resolution,
                            method="bicubic",
                        )
                    ),
                    tf.uint8,
                )

                # Distance matches Dataset._add_distances: T - range(T).
                T = tf.shape(wrist)[0]
                distance = T - tf.range(T, dtype=tf.int32)

                return TrajectoryObservation(
                    wrist_image=wrist.numpy(),
                    shoulder_image=shoulder.numpy(),
                    language_instruction=instruction.numpy(),
                    action_joint_8d=traj["action/joint_velocity"].numpy(),
                    image=image.numpy(),
                    distance=distance.numpy(),
                    tfrecord_path=str(tfrecord_path),
                    tfrecord_file_index=tfrecord_file_index,
                    tfrecord_example_index=current_example_index,
                    tfrecord_shard_id=tfrecord_shard_id,
                )

            current_example_index += 1

        raise ValueError(
            f"Failed to sample a trajectory with len >= {min_traj_len} "
            f"from {tfrecord_path}. Last observed len={last_wrist_len} "
            f"(starting example_index={example_index}, max_attempts={max_attempts})."
        )

    def _decode_jpeg_sequence(self, jpeg_sparse: tf.SparseTensor) -> tf.Tensor:
        # Sparse list of JPEG-encoded frames -> [T, 288, 512, 3] uint8.
        jpegs = tf.sparse.to_dense(jpeg_sparse, default_value=b"")
        # If the example has 0 frames for this key, `tf.map_fn` can otherwise
        # infer an empty tensor with rank 1, which then fails `ensure_shape`.
        out_spec = tf.TensorSpec(shape=(288, 512, 3), dtype=tf.uint8)
        frames = tf.map_fn(
            lambda b: tf.image.decode_jpeg(b, channels=3),
            jpegs,
            fn_output_signature=out_spec,
            parallel_iterations=32,
            infer_shape=True,
        )
        return tf.ensure_shape(frames, [None, 288, 512, 3])

    def _decode_trajectory(self, serialized_example: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Decode a single TFRecord example into image, language, and action sequences.

        Does not read `action/cartesian_velocity` (ignored for inference replay).

        Returns:
          traj dict with `action/joint_velocity` as [T, 8] after normalization + gripper.
        """
        feature_spec = {
            **{k: tf.io.VarLenFeature(tf.string) for k in self.IMG_KEYS},
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.F32_KEYS},
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.STR_KEYS},
        }

        parsed = tf.io.parse_single_example(serialized_example, feature_spec)

        out: Dict[str, tf.Tensor] = {}

        # Decode image sequences.
        for k in self.IMG_KEYS:
            out[k] = self._decode_jpeg_sequence(parsed[k])  # [T, 288, 512, 3]

        # Parse serialized float tensors (joint + gripper).
        for k in self.F32_KEYS:
            out[k] = tf.io.parse_tensor(parsed[k], out_type=tf.float32)
            if k == "action/gripper_position":
                out[k] = tf.reshape(out[k], [-1])
            else:
                out[k] = tf.ensure_shape(out[k], [None, None])

        # Strings: scalar.
        for k in self.STR_KEYS:
            out[k] = parsed[k]  # scalar tf.string

        # Ensure the action sequence length matches the image sequence length.
        T = tf.shape(out[self.IMG_KEYS[0]])[0]
        for k in self.F32_KEYS:
            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    tf.shape(out[k])[0], T, message=f"{k} length != image T"
                )
            ]):
                out[k] = tf.identity(out[k])

        # Same order as bageljax.data.dataset.Dataset._decode_example: gripper
        # binarization, then (jv - mean) / (std + eps) on the first 7 dims, then concat.
        grip_bin = binarize_gripper_action(out["action/gripper_position"])
        del out["action/gripper_position"]
        jv = out["action/joint_velocity"]
        jv = normalize_joint_velocity_7d(jv, self._jv_mean, self._jv_std, self._jv_eps)
        out["action/joint_velocity"] = tf.concat([jv, grip_bin[:, tf.newaxis]], axis=-1)

        # Repeat language strings to shape [T].
        T = tf.shape(out[self.IMG_KEYS[0]])[0]
        for k in self.STR_KEYS:
            out[k] = tf.fill([T], out[k])

        return out