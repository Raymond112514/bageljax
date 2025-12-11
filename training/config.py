import os
from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_config = dict(
        num_steps=int(1001000),
        log_interval=100,
        save_interval=20000,
        #eval_interval=5000, # we'll train without a validation set
        save_dir="gs://pranav-us-west1/log",
        resume_path=None,
        pretrained_bagel_path="gs://pranav-us-west1/worldmodelrl_starting_components/value_function_bagel_init/bagel",
        tokenizer_load_path="/nfs/nfs5/users/pranav/bagel_tokenizer",
        seed=137,
        #num_val_batches=16, # we'll train without a validation set
    )

    base_data_config = dict(
        data_paths=["gs://pranav-us-west1/datasets/droid/success/*.tfrecord"],
        batch_size=4,
        shuffle_buffer_size=1000,
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
