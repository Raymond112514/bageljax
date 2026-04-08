import os
from ml_collections import ConfigDict
import numpy as np

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
    )

    base_data_config = dict(
        data_paths=["gs://raymond-us-west1/droid/success/*.tfrecord"],
        batch_size=12,
        shuffle_buffer_size=100000,
        num_parallel_calls=10,
        max_prompt_length=254, # accounts for num tokens in longest prompt, the rewriting of the instruction, and the pad tokens needed to make global seq len a multiple of 128
    )

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
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_config,
            )
        ),
    }

    return possible_structures[config_string]
