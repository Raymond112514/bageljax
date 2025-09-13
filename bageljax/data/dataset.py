import fnmatch
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        if len(filtered_paths) == 0:
            print("Warning: glob_to_path_list didn't find any paths")
        path_list += filtered_paths
    return path_list

@tf.function
def binarize_gripper_action(x):
    # Reverse the tensor to process it from the end to the start
    x_reversed = tf.reverse(x, axis=[0])
    n = tf.shape(x_reversed)[0]

    # Compute the starts where the current element is zero and the next is non-zero
    x_reversed_padded = tf.concat([x_reversed[0:1], x_reversed[:-1]], axis=0)
    starts = tf.cast(
        (x_reversed_padded == 0.0) & (x_reversed != 0.0),
        tf.float32
    )

    # Initialize variables for the loop
    idx = tf.constant(0)
    in_sequence = tf.constant(0.0)
    prev_x = x_reversed[0]
    mask_ta = tf.TensorArray(dtype=tf.float32, size=n)

    def cond(idx, in_sequence, prev_x, mask_ta):
        return idx < n

    def body(idx, in_sequence, prev_x, mask_ta):
        start_flag = starts[idx]
        x_curr = x_reversed[idx]

        # Start a new sequence if start_flag == 1
        in_sequence = tf.where(start_flag > 0, 1.0, in_sequence)

        # If in_sequence is True and prev_x >= x_curr, we end the sequence
        in_sequence = tf.where(
            (in_sequence > 0) & (prev_x >= x_curr),
            0.0,
            in_sequence
        )

        # The mask is 0 where in_sequence is True, else 1
        mask_value = tf.where(in_sequence > 0, 0.0, 1.0)

        # Write the mask value
        mask_ta = mask_ta.write(idx, mask_value)

        idx += 1
        prev_x = x_curr

        return idx, in_sequence, prev_x, mask_ta

    # Run the loop
    idx, in_sequence, prev_x, mask_ta = tf.while_loop(
        cond, body, loop_vars=[idx, in_sequence, prev_x, mask_ta]
    )

    # Stack the mask and apply it
    mask = mask_ta.stack()
    x_zeroed = x_reversed * mask

    # Reverse the result to match the original order
    result = tf.reverse(x_zeroed, axis=[0])

    # Set all nonzero values to 1
    result = tf.where(result > 0, 1.0, 0.0)

    return result

class Dataset:
    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_proprio_metadata: Optional[dict] = None,
        normalization_type: Optional[str] = "normal",
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 10,
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        chunk_size: int = 16,
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

        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        self.is_train = train
        self.chunk_size = chunk_size
        self.num_parallel_calls = num_parallel_calls

        # construct a dataset for each sub-list of paths
        datasets = []
        for i, sub_data_paths in enumerate(data_paths):
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        # To allow for a large enough training shuffle buffer, we will not create one for validation
        if train:
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                )

        if train:
            # repeat the datasets
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard, and for validation as 
        # well it ensures the number of batches we sample is less than the validation size
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=True
        )

        opts = tf.data.Options()
        opts.autotune.enabled = True
        opts.experimental_deterministic = not self.is_train
        dataset = dataset.with_options(opts)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=self.num_parallel_calls,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=self.num_parallel_calls)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=self.num_parallel_calls)

        # yields trajectories
        #dataset = dataset.filter(self._filter_by_len)
        
        # # yields trajectories
        #dataset = dataset.map(
        #    self._process_actions, num_parallel_calls=self.num_parallel_calls
        #)

        # yields trajectories
        #dataset = dataset.map(self._chunk, num_parallel_calls=self.num_parallel_calls)

        # # yields trajectories
        #dataset = dataset.map(self._remove_unwanted_keys, num_parallel_calls=self.num_parallel_calls)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        # yields individual transitions
        #dataset = dataset.map(
        #    self._process_images, num_parallel_calls=self.num_parallel_calls
        #)

        # yields individual transitions
        #dataset = dataset.filter(self._lang_filter_fn)

        # To ensure determinism if we're in validation mode
        opts = tf.data.Options()
        opts.autotune.enabled = True
        opts.experimental_deterministic = not self.is_train
        dataset = dataset.with_options(opts)

        return dataset

    IMG_KEYS = (
        "wrist_image", 
        "shoulder_image_1", 
        "shoulder_image_2",
    )
    F32_KEYS = (
        "action/cartesian_velocity",
        "observation/robot_state/gripper_position",
        "observation/robot_state/joint_positions",
    )
    STR_KEYS = (
        "language_instruction1",
        "language_instruction2",
        "language_instruction3",
    )

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
                parallel_iterations=self.num_parallel_calls,
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
            if "gripper" in k:
                out[k] = tf.ensure_shape(out[k], [None,])
            else:
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

    ######################### Trajectory level transforms ############################
    
    def _filter_by_len(self, traj):
        traj_len = tf.shape(traj["observation/robot_state/joint_positions"])[0]
        # Return a boolean indicating whether to keep this example
        return traj_len >= 20

    def _process_actions(self, traj):
        # normalize actions and proprio
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # normalize to mean 0, std 1
                action_mean, action_std = self.action_proprio_metadata["action_mean"], self.action_proprio_metadata["action_std"]
                proprio_mean, proprio_std = self.action_proprio_metadata["proprio_mean"], self.action_proprio_metadata["proprio_std"]
                traj["actions"] = tf.concat([(
                    traj["actions"][:, :7] - action_mean
                    ) / action_std, traj["actions"][:, 7:]], axis=1)
                traj["observations"]["proprio"] = tf.concat([(
                    traj["observations"]["proprio"][:, :7] - proprio_mean
                    ) / proprio_std, traj["observations"]["proprio"][:, 7:]], axis=1)
            else:
                raise ValueError
            
        # binarize gripper component of actions
        gripper_actions = traj["actions"][:, -1]
        binarized_gripper_actions = binarize_gripper_action(gripper_actions)
        traj["actions"] = tf.concat([traj["actions"][:, :-1], binarized_gripper_actions[..., tf.newaxis]], axis=1)

        return traj

    def _chunk(self, traj):
        traj_len = tf.shape(traj["actions"])[0]
        actions_remaining = traj_len - tf.range(traj_len)
        
        def collect_and_pad_sequences(idx_length):
            idx, length = idx_length[0], idx_length[1]
            sequence = tf.gather(traj["actions"], tf.range(idx, idx+length))
            
            # Pad with self.chunk_size additional zeros
            zero_movement_actions = tf.zeros((self.chunk_size, 7), dtype=tf.float32)
            zero_movement_actions = (zero_movement_actions - self.action_proprio_metadata["action_mean"]) / self.action_proprio_metadata["action_std"]
            zero_actions = tf.concat([zero_movement_actions, tf.tile(sequence[-1:, -1:], (self.chunk_size, 1))], axis=1)
            sequence = tf.concat([sequence, zero_actions], axis=0)

            # Select the first self.chunk_size actions
            sequence = sequence[:self.chunk_size]

            return sequence
        
        indices_and_lengths = tf.stack([tf.range(traj_len), actions_remaining], axis=1)

        action_sequences = tf.map_fn(
            collect_and_pad_sequences, 
            indices_and_lengths, 
            dtype=tf.float32,
            parallel_iterations=self.num_parallel_calls  # Adjust this based on your hardware capabilities
        )

        traj["action_sequences"] = action_sequences

        return traj

    ######################### Example level transforms ############################

    def _lang_filter_fn(self, batch_elem):
        keep = batch_elem["language"] != b""
        return keep
    
    def _process_images(self, traj):
        # Resize images to (384, 384) and scale to [-1, 1]
        traj["observations"]["shoulder_1"] = tf.ensure_shape(traj["observations"]["shoulder_1"], [None, 180, 320, 3])
        traj["observations"]["shoulder_2"] = tf.ensure_shape(traj["observations"]["shoulder_2"], [None, 180, 320, 3])
        traj["observations"]["wrist"] = tf.ensure_shape(traj["observations"]["wrist"], [None, 180, 320, 3])

        traj["observations"]["shoulder_1"] = tf.cast(tf.image.resize(traj["observations"]["shoulder_1"], (384, 384), method="bicubic"), tf.uint8)
        traj["observations"]["shoulder_2"] = tf.cast(tf.image.resize(traj["observations"]["shoulder_2"], (384, 384), method="bicubic"), tf.uint8)
        traj["observations"]["wrist"] = tf.cast(tf.image.resize(traj["observations"]["wrist"], (384, 384), method="bicubic"), tf.uint8)

        return traj

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()