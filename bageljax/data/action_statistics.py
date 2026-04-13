"""
Global normalization stats for `action/joint_velocity` (7D), computed over
Droid success TFRecords (see load_data.py). Gripper (8th dim) is not normalized.
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
