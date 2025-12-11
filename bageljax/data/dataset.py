import fnmatch
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging

class Dataset:
    IMG_KEYS = (
        "wrist_image", 
        "shoulder_image_1", 
        "shoulder_image_2",
    )
    F32_KEYS = (
        "action/cartesian_velocity",
    )
    STR_KEYS = (
        "language_instruction1",
        "language_instruction2",
        "language_instruction3",
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
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to Dataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.is_train = bool(train)
        self.num_parallel_calls = num_parallel_calls
        # Integer parallelism for non-dataset APIs like tf.map_fn
        self._map_parallel_iterations = (
            int(num_parallel_calls) if isinstance(num_parallel_calls, int) else 32
        )
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        # Seed domains: keep different RNG uses separated for clarity.
        self._path_shuffle_base_seed = self.seed + 101
        self._subdataset_seed_stride = 1000
        self._mix_seed = self.seed + 202
        self._subdataset_shuffle_base_seed = self.seed + 303

        # Build one dataset per sub-dataset
        sub_datasets = []
        for i, sub_paths in enumerate(data_paths):
            # Construct the subdataset
            sub_ds = self._construct_tf_dataset(
                sub_paths,
                subdataset_index=i,
            )

            # Shuffle each subdataset
            if self.is_train and self.shuffle_buffer_size > 0:
                per_ds_buf = max(1, int(self.shuffle_buffer_size * float(sample_weights[i])))
                shuffle_seed = self._subdataset_shuffle_base_seed + self._subdataset_seed_stride * i
                sub_ds = sub_ds.shuffle(
                    buffer_size=per_ds_buf,
                    seed=shuffle_seed,
                    reshuffle_each_iteration=True,
                )

            # For training, repeat each sub-dataset so it never exhausts.
            if self.is_train:
                sub_ds = sub_ds.repeat()

            sub_datasets.append(sub_ds)

        if len(sub_datasets) == 1:
            dataset = sub_datasets[0]
        else:
            # for validation, we want to be able to iterate through the entire dataset;
            # for training, we want to make sure that no sub-dataset is ever exhausted
            # or the sampling ratios will be off. this should never happen because of the
            # repeat() above, but `stop_on_empty_dataset` is a safeguard
            dataset = tf.data.Dataset.sample_from_datasets(
                sub_datasets,
                weights=sample_weights,
                seed=self._mix_seed,
                stop_on_empty_dataset=True, 
            )

        # Batch; drop_remainder=True for stable shapes during training.
        dataset = dataset.batch(
            self.batch_size,
            num_parallel_calls=self.num_parallel_calls,
            drop_remainder=self.is_train,
            deterministic=not self.is_train,
        )

        # Dataset-level options: deterministic only for eval.
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

    def _construct_tf_dataset(self, paths: List[str], subdataset_index: int,) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of path/glob strings.
        The dataset yields a dictionary of tensors for each transition.
        """
        all_paths = self._expand_and_check_paths(paths)

        # Dataset of file paths. For training we shuffle + reshuffle each "epoch";
        # for validation we keep a fixed, sorted order.
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

        # Stream records from all shards in parallel.
        ds = tf.data.TFRecordDataset(
            path_ds,
            num_parallel_reads=self.num_parallel_calls,
        )

        # Trajectory-level pipeline -----------------------------------------
        ds = ds.map(
            self._decode_example,
            num_parallel_calls=self.num_parallel_calls,
        )
        ds = ds.filter(self._filter_by_len)
        ds = ds.map(
            self._add_distances,
            num_parallel_calls=self.num_parallel_calls,
        )

        # Unbatch to obtain individual transitions.
        ds = ds.unbatch()

        # Example-level pipeline --------------------------------------------
        ds = ds.map(
            self._select_language_instr,
            num_parallel_calls=self.num_parallel_calls,
        )
        ds = ds.filter(self._lang_filter_fn)
        ds = ds.map(
            self._stack_and_reshape_images,
            num_parallel_calls=self.num_parallel_calls,
        )

        # Determinism flag for validation.
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

        # Crash if the image list is absent or empty.
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

        # Enforce the known image size; will raise if any frame disagrees.
        frames = tf.ensure_shape(frames, [None, 288, 512, 3])
        return frames

    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        """
        Graph-friendly parser that enforces presence of all required fields.
        Returns TF tensors only:
        - images: uint8 [T, 288, 512, 3]
        - float tensors: parsed from serialized tensors (dtype float32)
        - strings: scalar tf.string
        """
        # For lists-of-bytes (JPEG frames) we must use VarLenFeature.
        # For required scalars, omit default_value -> parse_single_example errors if missing.
        feature_spec = {
            **{k: tf.io.VarLenFeature(tf.string) for k in self.IMG_KEYS},
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.F32_KEYS},  # required
            **{k: tf.io.FixedLenFeature([], tf.string) for k in self.STR_KEYS},  # required
        }

        parsed = tf.io.parse_single_example(serialized_example, feature_spec)
        out = {}

        # Decode image sequences
        for k in self.IMG_KEYS:
            out[k] = self._decode_jpeg_sequence(parsed[k])  # [T, 288, 512, 3] uint8

        # Parse serialized float tensors (required); dtype is float32.
        for k in self.F32_KEYS:
            out[k] = tf.io.parse_tensor(parsed[k], out_type=tf.float32)
            out[k] = tf.ensure_shape(out[k], [None, None])

        # Strings (required)
        for k in self.STR_KEYS:
            out[k] = parsed[k]  # scalar tf.string

        # --- sanity check: ensure time dim matches image T ---------------------------
        T = tf.shape(out[self.IMG_KEYS[0]])[0]
        for k in self.F32_KEYS:
            # Attach the assertion without replacing the tensor with an op.
            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    tf.shape(out[k])[0], T, message=f"{k} length != image T"
                )
            ]):
                out[k] = tf.identity(out[k])  # keeps it a Tensor

        # --- stop-action mask (use last axis, not hardcoded axis=1) ------------------
        a_cartesian_velocity = out["action/cartesian_velocity"]      # [T, D]
        vec3 = a_cartesian_velocity[..., :3]                         # [T, 3]
        xyz_action_se = tf.reduce_sum(tf.square(vec3), axis=-1)      # [T]
        mask = xyz_action_se > 1e-3                                  # [T] boolean

        # Apply mask on the time axis
        for k in self.IMG_KEYS:
            out[k] = tf.boolean_mask(out[k], mask, axis=0)
        for k in self.F32_KEYS:
            out[k] = tf.boolean_mask(out[k], mask, axis=0)

        # Drop now-unused key
        del out["action/cartesian_velocity"]

        T = tf.shape(out[self.IMG_KEYS[0]])[0]  # time length after mask
        # Repeat language strings to enable unbatching later
        for k in self.STR_KEYS:
            out[k] = tf.ensure_shape(out[k], [])
            out[k] = tf.fill([T], out[k])        # -> shape [T]

        return out
    
    def _filter_by_len(self, traj):
        traj_len = tf.shape(traj["wrist_image"])[0]
        # Return a boolean indicating whether to keep this example
        return traj_len >= 20
    
    def _add_distances(self, traj):
        T = tf.shape(traj["wrist_image"])[0]
        traj["distance"] = T - tf.range(T, dtype=tf.int32)
        return traj

    # --------------------------------------------------------------------- #
    # Example-level transforms                                              #
    # --------------------------------------------------------------------- #

    def _select_language_instr(self, transition):
        random_number = tf.random.uniform((1,), maxval=3, dtype=tf.int32)[0]
        language_1 = transition["language_instruction1"]
        language_2 = transition["language_instruction2"]
        language_3 = transition["language_instruction3"]
        
        language = language_1
        language = tf.where(random_number == 1, language_2, language)
        language = tf.where(random_number == 2, language_3, language)

        transition["language_instruction"] = language
        del transition["language_instruction1"]
        del transition["language_instruction2"]
        del transition["language_instruction3"]

        return transition

    def _lang_filter_fn(self, transition):
        keep = transition["language_instruction"] != b""
        return keep
    
    def _stack_and_reshape_images(self, transition):
        random_number = tf.random.uniform((1,), maxval=2, dtype=tf.int32)[0]
        shoulder = transition["shoulder_image_1"]
        shoulder = tf.where(random_number == 1, transition["shoulder_image_2"], shoulder)

        wrist = transition["wrist_image"]

        shoulder_and_wrist = tf.concat([shoulder, wrist], axis=0)
        shoulder_and_wrist = tf.ensure_shape(shoulder_and_wrist, [576, 512, 3])

        shoulder_and_wrist = tf.cast(tf.round(tf.image.resize(shoulder_and_wrist, (672, 560), method="bicubic")), tf.uint8)
        
        transition["image"] = shoulder_and_wrist
        del transition["shoulder_image_1"]
        del transition["shoulder_image_2"]
        del transition["wrist_image"]

        return transition

    ######################### Iterator ############################

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()