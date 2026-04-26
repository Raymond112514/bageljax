import os
from ml_collections import ConfigDict
import numpy as np

from bageljax.data.data_utils import (
    ACTION_JOINT_VELOCITY_MEAN,
    ACTION_JOINT_VELOCITY_STD,
    ACTION_NORM_EPS,
)

def get_config(config_string):
    base_config = dict(
        num_steps=int(1000000),
        log_interval=100,
        save_interval=1000,
        save_dir="gs://raymond-us-west1/value_function_logs",
        resume_path=None,
        pretrained_bagel_path="gs://raymond-us-west1/value_function_starting_components/bagel",
        tokenizer_load_path="/nfs/nfs5/users/raymond/bagel_tokenizer",
        seed=137,
        # If True, each step mixes half-batch DROID + half-batch RoboArena
        use_roboarena=True,
    )

    base_data_config = dict(
        data_paths=["gs://raymond-us-west1/droid/success/*.tfrecord"],
        batch_size=12,
        shuffle_buffer_size=100000,
        num_parallel_calls=10,
        action_chunk_size=30,
        max_prompt_length=224, # accounts for num tokens in longest prompt, the rewriting of the instruction, and the pad tokens needed to make global seq len a multiple of 128
        # Normalization for first 7 dims of 8D action (joint velocity); gripper stays ±1.
        action_joint_velocity_mean=list(ACTION_JOINT_VELOCITY_MEAN),
        action_joint_velocity_std=list(ACTION_JOINT_VELOCITY_STD),
        action_norm_eps=ACTION_NORM_EPS,
    )

    _ROBOARENA_PREFIX = "gs://raymond-us-west1/droid_labeled/roboarena_renamed"
    _ROBOARENA_SHARD_BASENAME = "roboarena"
    _roboarena_shard_paths = [
        f"{_ROBOARENA_PREFIX}/{_ROBOARENA_SHARD_BASENAME}-{i:05d}.tfrecord"
        for i in range(40)
    ]

    roboarena_data_config = {
        **base_data_config,
        "data_paths": [_roboarena_shard_paths],
    }

    possible_structures = {
        "bagel_value_function": ConfigDict(
            dict(
                policy_kwargs=dict(
                    num_buckets=64,
                    discount_factor=0.993,
                    learning_rate=1e-4, 
                    weight_decay=0.0,
                    b2=0.95,
                    eps=1e-15,
                    decay_steps=None,
                    warmup_steps=2500,
                    target_update_rate=0.9999,
                    obs_dropout_prob=0.5, # Randomly drop out the image tokens with probability p
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                roboarena_dataset_kwargs=dict(
                    **roboarena_data_config,
                ),
                **base_config,
            )
        ),
    }

    return possible_structures[config_string]
