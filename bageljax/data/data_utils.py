"""
Shared data utilities for action normalization and gripper preprocessing.
"""

from __future__ import annotations

from typing import Union

import tensorflow as tf

# Population mean / std over ~19.5M timesteps (72124 episodes).
ACTION_JOINT_VELOCITY_MEAN: tuple[float, ...] = (
    -0.0038499487288812697,
    0.0218246295600785,
    0.00191047289186909,
    0.03473553986227862,
    0.001975067213626195,
    0.0018849006539698974,
    0.004859663189535178,
)

ACTION_JOINT_VELOCITY_STD: tuple[float, ...] = (
    0.15344765590677564,
    0.30208207131165793,
    0.14993731510508318,
    0.29458300381611563,
    0.21809917012006227,
    0.24094136950683148,
    0.2598872753694674,
)

ACTION_NORM_EPS: float = 1e-8


@tf.function
def binarize_gripper_action(x):
    """Full-trajectory gripper channel [T] -> binarized {-1, +1}.

    Safe for T==0 (e.g. stop-action mask removed every timestep).
    """
    x = tf.reshape(x, [-1])
    n = tf.shape(x)[0]

    def nonempty():
        # Reverse the tensor to process it from the end to the start.
        x_reversed = tf.reverse(x, axis=[0])

        # Compute the starts where the current element is zero and the next is non-zero.
        x_reversed_padded = tf.concat([x_reversed[0:1], x_reversed[:-1]], axis=0)
        starts = tf.cast(
            (x_reversed_padded == 0.0) & (x_reversed != 0.0),
            tf.float32,
        )

        idx = tf.constant(0)
        in_sequence = tf.constant(0.0)
        prev_x = x_reversed[0]
        mask_ta = tf.TensorArray(dtype=tf.float32, size=n)

        def cond(idx, in_sequence, prev_x, mask_ta):
            return idx < n

        def body(idx, in_sequence, prev_x, mask_ta):
            start_flag = starts[idx]
            x_curr = x_reversed[idx]

            in_sequence = tf.where(start_flag > 0, 1.0, in_sequence)

            in_sequence = tf.where(
                (in_sequence > 0) & (prev_x >= x_curr),
                0.0,
                in_sequence,
            )

            mask_value = tf.where(in_sequence > 0, 0.0, 1.0)

            mask_ta = mask_ta.write(idx, mask_value)

            idx += 1
            prev_x = x_curr

            return idx, in_sequence, prev_x, mask_ta

        idx, in_sequence, prev_x, mask_ta = tf.while_loop(
            cond, body, loop_vars=[idx, in_sequence, prev_x, mask_ta]
        )

        mask = mask_ta.stack()
        x_zeroed = x_reversed * mask

        result = tf.reverse(x_zeroed, axis=[0])

        result = tf.where(result > 0, 1.0, -1.0)

        return result

    return tf.cond(
        tf.equal(n, 0),
        lambda: tf.zeros([0], dtype=tf.float32),
        nonempty,
    )


def normalize_joint_velocity_7d(
    jv: tf.Tensor,
    mean7: tf.Tensor,
    std7: tf.Tensor,
    eps: Union[tf.Tensor, float] = ACTION_NORM_EPS,
) -> tf.Tensor:
    """(jv - mean) / (std + eps), jv shape [T, 7], mean/std shape [7]."""
    if not isinstance(eps, tf.Tensor):
        eps = tf.constant(eps, dtype=tf.float32)
    return (jv - mean7[tf.newaxis, :]) / (std7[tf.newaxis, :] + eps)


def default_joint_velocity_norm_tensors() -> tuple[tf.Tensor, tf.Tensor]:
    """TF constants [7] for use in tf.data graphs."""
    m = tf.constant(ACTION_JOINT_VELOCITY_MEAN, dtype=tf.float32)
    s = tf.constant(ACTION_JOINT_VELOCITY_STD, dtype=tf.float32)
    return m, s
