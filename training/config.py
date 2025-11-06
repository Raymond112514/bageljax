import os
from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_config = dict(
        num_steps=int(2001000),
        log_interval=100,
        save_interval=10000,
        #eval_interval=5000, # we'll train without a validation set
        save_dir="gs://pranav-us-west1/log",
        resume_path=None,
        pretrained_bagel_path="gs://pranav-europe-west4/worldmodelrl_starting_components/half_bagel_weights/bagel",
        action_tokenizer_resume_path="gs://pranav-europe-west4/worldmodelrl_starting_components/act_tok_chunk_16_enc_only_ckpt",
        tokenizer_load_path="/nfs/nfs5/users/pranav/bagel_tokenizer",
        seed=137,
        #num_val_batches=16, # we'll train without a validation set
    )

    base_data_config = dict(
        dataset_path="gs://pranav-us-west1/datasets/droid/success", 
        batch_size=4,
        shuffle_buffer_size=1000,
        chunk_size=16,
        num_parallel_calls=10,
        action_proprio_metadata=dict(
            mean=np.array([0.011434038169682026, 0.2440052479505539, -0.013901660218834877, -2.0293116569519043, -0.03873773291707039, 2.3191726207733154, 0.0831032246351242, -0.09849374741315842,], dtype=np.float32),
            std=np.array([0.31484323740005493, 0.5151926875114441, 0.2791784405708313, 0.5057438611984253, 0.5162752270698547, 0.4621083438396454, 0.7458285093307495, 0.9951376914978027,], dtype=np.float32),
        ),
        max_prompt_length=226, # used to be 120; tokens, corresponds to the max number of tokens needed for any of the language instructions, not including bos/eos
        action_tokens_offset=150000, # first special token is at 151644, vocab size is 152064
    )

    possible_structures = {
        "bagelvla": ConfigDict(
            dict(
                policy_kwargs=dict(
                    learning_rate=1e-4, 
                    weight_decay=0.0,
                    b2=0.95,
                    eps=1e-15,
                    decay_steps=None,
                    warmup_steps=5000,
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
