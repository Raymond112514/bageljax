"""Shared dataset class for any robot data that stores a pre-computed (T, 10)
reward-bin distribution alongside pre-chunked actions.  Used for both DROID
and RoboArena trajectories.
"""
import fnmatch
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from absl import logging

from bageljax.data.data_utils import binarize_gripper_action, normalize_joint_velocity_7d

# Default pre-chunk sizes (overridable per dataset via constructor).
_DEFAULT_PRE_CHUNK_SIZE = 10
_DEFAULT_NUM_REWARD_BINS = 10

def print_green(message):
    print(f"\033[92m{message}\033[0m")

class ValueDataset:
    IMG_KEYS = (
        "shoulder_image_1",
    )
    F32_SERIALIZED_KEYS = (
        "action/joint_velocity",
        "action/gripper_position",
        "rewards",  # serialized float32 tensor of shape (T, 10) — bin distribution
    )
    # Used only for the time-dim sanity check (checks axis-0 == T).
    F32_KEYS = (
        "action/joint_velocity",
        "action/gripper_position",
        "rewards",
    )
    STR_KEYS = (
        "language_instruction1",
    )

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 10,
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        num_parallel_calls: int = 10,
        action_chunk_size: int = 30,
        pre_chunk_size: int = _DEFAULT_PRE_CHUNK_SIZE,
        num_reward_bins: int = _DEFAULT_NUM_REWARD_BINS,
        action_joint_velocity_mean: Optional[Sequence[float]] = None,
        action_joint_velocity_std: Optional[Sequence[float]] = None,
        action_norm_eps: float = 1e-8,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to ValueDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.is_train = bool(train)
        self.num_parallel_calls = num_parallel_calls
        self._map_parallel_iterations = (
            int(num_parallel_calls) if isinstance(num_parallel_calls, int) else 32
        )
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.action_chunk_size = int(action_chunk_size)
        self._pre_chunk_size = int(pre_chunk_size)
        self._num_reward_bins = int(num_reward_bins)

        if action_joint_velocity_mean is not None and action_joint_velocity_std is not None:
            mean = np.asarray(action_joint_velocity_mean, dtype=np.float32)
            std = np.asarray(action_joint_velocity_std, dtype=np.float32)
            if mean.shape != (7,) or std.shape != (7,):
                raise ValueError(
                    "action_joint_velocity_mean/std must have shape (7,), got "
                    f"{mean.shape}, {std.shape}",
                )
            self._jv_mean = tf.constant(mean)
            self._jv_std = tf.constant(std)
            self._jv_eps = tf.constant(float(action_norm_eps), tf.float32)
            self._normalize_joint_velocity = True
        else:
            self._normalize_joint_velocity = False
            self._jv_mean = None
            self._jv_std = None
            self._jv_eps = None

        # Seed domains: keep different RNG uses separated for clarity.
        self._path_shuffle_base_seed = self.seed + 101
        self._subdataset_seed_stride = 1000
        self._mix_seed = self.seed + 202
        self._subdataset_shuffle_base_seed = self.seed + 303

        sub_datasets = []
        for i, sub_paths in enumerate(data_paths):
            sub_ds = self._construct_tf_dataset(sub_paths, subdataset_index=i)

            if self.is_train and self.shuffle_buffer_size > 0:
                per_ds_buf = max(1, int(self.shuffle_buffer_size * float(sample_weights[i])))
                shuffle_seed = self._subdataset_shuffle_base_seed + self._subdataset_seed_stride * i
                sub_ds = sub_ds.shuffle(
                    buffer_size=per_ds_buf,
                    seed=shuffle_seed,
                    reshuffle_each_iteration=True,
                )

            if self.is_train:
                sub_ds = sub_ds.repeat()

            sub_datasets.append(sub_ds)

        if len(sub_datasets) == 1:
            dataset = sub_datasets[0]
        else:
            dataset = tf.data.Dataset.sample_from_datasets(
                sub_datasets,
                weights=sample_weights,
                seed=self._mix_seed,
                stop_on_empty_dataset=True,
            )

        dataset = dataset.batch(
            self.batch_size,
            num_parallel_calls=self.num_parallel_calls,
            drop_remainder=self.is_train,
            deterministic=not self.is_train,
        )

        opts = tf.data.Options()
        opts.autotune.enabled = True
        opts.experimental_deterministic = not self.is_train
        dataset = dataset.with_options(opts)

        self.tf_dataset = dataset

    # --------------------------------------------------------------------- #
    # Internal construction of a single (sub-)dataset                       #
    # --------------------------------------------------------------------- #

    def _expand_and_check_paths(self, sub_paths: List[str]) -> List[str]:
        """Expand glob patterns, dedup, and sort for stability."""
        all_paths: List[str] = []
        for p in sub_paths:
            matched = tf.io.gfile.glob(p)
            if not matched:
                logging.warning("No files match pattern/path: %s", p)
            all_paths.extend(matched)

        if not all_paths:
            raise ValueError("No TFRecord files found in data_paths.")

        all_paths = sorted(set(all_paths))
        return all_paths

    def _construct_tf_dataset(self, paths: List[str], subdataset_index: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of path/glob strings.
        The dataset yields a dictionary of tensors for each transition.
        """
        all_paths = self._expand_and_check_paths(paths)

        path_ds = tf.data.Dataset.from_tensor_slices(all_paths)
        if self.is_train and len(all_paths) > 1:
            path_seed = self._path_shuffle_base_seed + (
                self._subdataset_seed_stride * subdataset_index
            )
            path_ds = path_ds.shuffle(
                buffer_size=len(all_paths),
                seed=path_seed,
                reshuffle_each_iteration=True,
            )

        ds = tf.data.TFRecordDataset(
            path_ds,
            num_parallel_reads=self.num_parallel_calls,
        )

        # Trajectory-level pipeline
        ds = ds.map(self._decode_example, num_parallel_calls=self.num_parallel_calls)
        ds = ds.filter(self._filter_by_len)
        ds = ds.map(self._add_next_step_reward_targets, num_parallel_calls=self.num_parallel_calls)
        ds = ds.map(self._add_action_chunks, num_parallel_calls=self.num_parallel_calls)
        
        # Unbatch to individual transitions
        ds = ds.unbatch()

        # Example-level pipeline
        ds = ds.map(self._select_language_instr, num_parallel_calls=self.num_parallel_calls)
        ds = ds.filter(self._lang_filter_fn)
        ds = ds.map(self._stack_and_reshape_images, num_parallel_calls=self.num_parallel_calls)
        opts = tf.data.Options()
        opts.autotune.enabled = True
        opts.experimental_deterministic = not self.is_train
        ds = ds.with_options(opts)

        return ds

    # --------------------------------------------------------------------- #
    # Decoding / trajectory-level transforms                                #
    # --------------------------------------------------------------------- #

    def _decode_jpeg_sequence(self, jpeg_sparse: tf.SparseTensor) -> tf.Tensor:
        """Sparse list of JPEG-encoded frames -> [T, 288, 512, 3] uint8."""
        jpegs = tf.sparse.to_dense(jpeg_sparse, default_value=b"")

        with tf.control_dependencies([
            tf.debugging.assert_greater(
                tf.shape(jpegs)[0], 0, message="No JPEG frames found for an image key."
            )
        ]):
            frames = tf.map_fn(
                lambda b: tf.image.decode_jpeg(b, channels=3),
                jpegs,
                fn_output_signature=tf.uint8,
                parallel_iterations=self._map_parallel_iterations,
                infer_shape=True,
            )

        frames = tf.ensure_shape(frames, [None, 288, 512, 3])
        return frames

    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        """
        Graph-friendly parser that enforces presence of all required fields.
        Returns TF tensors only:
        - images:   uint8  [T, 288, 512, 3]
        - float tensors: parsed from serialized tensors (float32)
        - rewards:  float32 [T, 10] — pre-computed reward bin distribution
        - strings:  scalar tf.string
        """
        feature_spec = {
            **{k: tf.io.VarLenFeature(tf.string) for k in self.IMG_KEYS},
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.F32_SERIALIZED_KEYS},
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.STR_KEYS},
        }

        parsed = tf.io.parse_single_example(serialized_example, feature_spec)
        out = {}

        for k in self.IMG_KEYS:
            out[k] = self._decode_jpeg_sequence(parsed[k])  # [T, 288, 512, 3] uint8

        for k in self.F32_SERIALIZED_KEYS:
            out[k] = tf.io.parse_tensor(parsed[k], out_type=tf.float32)
            if k == "action/joint_velocity":
                out[k] = tf.ensure_shape(out[k], [None, self._pre_chunk_size, 7])
            elif k == "rewards":
                out[k] = tf.ensure_shape(out[k], [None, self._num_reward_bins])
            else:
                out[k] = tf.ensure_shape(out[k], [None, self._pre_chunk_size])

        for k in self.STR_KEYS:
            out[k] = parsed[k]  # scalar tf.string

        # Sanity check: time dim of every float tensor must equal image T.
        T = tf.shape(out[self.IMG_KEYS[0]])[0]
        for k in self.F32_KEYS:
            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    tf.shape(out[k])[0], T, message=f"{k} length != image T"
                )
            ]):
                out[k] = tf.identity(out[k])

        # 7D velocity + 1D binarized gripper -> 8D action.
        grip_bin = binarize_gripper_action(out["action/gripper_position"])
        grip_bin = tf.reshape(grip_bin, tf.stack([T, tf.constant(self._pre_chunk_size)]))
        del out["action/gripper_position"]
        jv = out["action/joint_velocity"]  # [T, pre_chunk_size, 7]
        if self._normalize_joint_velocity:
            jv = normalize_joint_velocity_7d(jv, self._jv_mean, self._jv_std, self._jv_eps)
        out["action/joint_velocity"] = tf.concat(
            [jv, grip_bin[:, :, tf.newaxis]], axis=-1
        )  # [T, PRE_CHUNK_SIZE, 8]

        T = tf.shape(out[self.IMG_KEYS[0]])[0]
        for k in self.STR_KEYS:
            out[k] = tf.ensure_shape(out[k], [])
            out[k] = tf.fill([T], out[k])  # -> shape [T]

        return out

    def _filter_by_len(self, traj):
        traj_len = tf.shape(traj[self.IMG_KEYS[0]])[0]
        k_outer = self.action_chunk_size // self._pre_chunk_size
        return traj_len >= k_outer + 1

    def _add_next_step_reward_targets(self, traj):
        """Align value prediction target r_{i+1} with (s_i, a_i)."""
        T = tf.shape(traj[self.IMG_KEYS[0]])[0]
        with tf.control_dependencies([
            tf.debugging.assert_greater_equal(
                T, 2,
                message="Trajectory length must be >= 2 to pair (s_i, a_i) with r_{i+1}.",
            )
        ]):
            tmax = T - 1
        for k in self.IMG_KEYS:
            traj[k] = traj[k][:tmax]
        traj["action/joint_velocity"] = traj["action/joint_velocity"][:tmax]
        rewards = traj["rewards"]
        traj["value_target"] = rewards[1:]  # shape [T-1, 10]
        del traj["rewards"]
        for k in self.STR_KEYS:
            traj[k] = traj[k][:tmax]

        T = tf.shape(traj[self.IMG_KEYS[0]])[0]
        for k, v in traj.items():
            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    tf.shape(v)[0], T, message=f"{k}: time dim != image time dim"
                )
            ]):
                traj[k] = tf.identity(v)
        return traj

    def _add_action_chunks(self, traj):
        """Build fixed-length action windows for pre-chunked actions.

        For window start t with outer blocks a_t..a_{t+k_outer-1}, align the
        value target with r_{t+k_outer} (reward after the last block), not
        r_{t+1}. With value_target[j]=r_{j+1} from _add_next_step_reward_targets,
        that is value_target[t + k_outer - 1] per row.
        """
        micro_k = int(self.action_chunk_size)
        if micro_k % self._pre_chunk_size != 0:
            raise ValueError(
                f"action_chunk_size ({micro_k}) must be divisible by pre_chunk_size "
                f"({self._pre_chunk_size})."
            )
        k_outer = micro_k // self._pre_chunk_size

        actions = traj["action/joint_velocity"]  # [T, pre_chunk_size, 8]
        T = tf.shape(actions)[0]

        valid_T = T - k_outer + 1
        base = tf.range(valid_T, dtype=tf.int32)[:, None]
        offsets = tf.range(k_outer, dtype=tf.int32)[None, :]
        idx = base + offsets  # [valid_T, k_outer]

        gathered = tf.gather(actions, idx, axis=0)  # [valid_T, k_outer, PRE_CHUNK_SIZE, 8]
        action_chunks = tf.reshape(
            gathered,
            tf.stack([valid_T, tf.constant(micro_k, dtype=tf.int32), tf.constant(8, dtype=tf.int32)]),
        )  # [valid_T, micro_k, 8]

        for k in list(traj.keys()):
            if k == "action/joint_velocity":
                continue
            traj[k] = traj[k][:valid_T]

        traj["action/joint_velocity_chunk"] = tf.ensure_shape(action_chunks, [None, micro_k, 8])
        del traj["action/joint_velocity"]
        return traj

    # --------------------------------------------------------------------- #
    # Example-level transforms                                              #
    # --------------------------------------------------------------------- #

    def _select_language_instr(self, transition):
        language = transition["language_instruction1"]
        transition["language_instruction"] = language
        del transition["language_instruction1"]
        return transition

    def _lang_filter_fn(self, transition):
        return transition["language_instruction"] != b""

    def _stack_and_reshape_images(self, transition):
        shoulder = transition["shoulder_image_1"]
        shoulder = tf.ensure_shape(shoulder, [288, 512, 3])
        shoulder = tf.cast(
            tf.round(tf.image.resize(shoulder, (672, 560), method="bicubic")),
            tf.uint8,
        )
        transition["image"] = shoulder
        del transition["shoulder_image_1"]
        return transition

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
