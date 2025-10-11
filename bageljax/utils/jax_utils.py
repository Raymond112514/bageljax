import logging
import os
from typing import Any, Optional, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental import multihost_utils, mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental.multihost_utils import process_allgather
import flax
import optax
import numpy as np
import tqdm
import traceback
import wandb
from flax.training import checkpoints
import os
import random
import copy
import datetime


def host_broadcast_str(x: str) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    multihost_utils.assert_equal(
        len(x), f"String lengths are not equal: got {len(x)} for {jax.process_index()}"
    )
    encoded = np.array([ord(c) for c in x], dtype=np.uint8)
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    return "".join([chr(u) for u in encoded])


def initialize_compilation_cache(
    cache_dir=os.path.expanduser("~/.jax_compilation_cache"),
):
    """Initializes the Jax persistent compilation cache."""
    pass

    # Right now this function doesn't do anything, but when you implement it, the thing to 
    # do for multi-host training is to set the compilation cache dir to a google bucket which 
    # all workers can see


def create_sharding(shard_type, train_state_shape=None):
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('devices'))
    data_sharding = NamedSharding(mesh, PartitionSpec('devices'))
    no_shard = NamedSharding(mesh, PartitionSpec())
    num_hosts = jax.device_count() // len(jax.local_devices())

    if shard_type == 'dp':
        # Data-Parallelism.
        # - A full copy of params are on each device.
        # - Each device gets an independent slice of the batch.
        train_state_sharding = no_shard
    elif shard_type == 'fsdp':
        # Fully-Sharded Data Parallism.
        # - Each device gets an independent slice of the batch.
        # - Parameters are sharded among each device, along the largest axis.
        def shard_parameter(param):
            shape = param.shape
            all_nones = (None,) * param.ndim
            min_size_to_shard_mb = 4
            if np.prod(shape) * param.dtype.itemsize <= min_size_to_shard_mb * (2 ** 20):
                return all_nones
            idx = np.argsort(shape)[::-1]
            for i in idx:
                if shape[i] % jax.device_count() == 0:
                    return all_nones[:i] + ('devices',) + all_nones[i+1:]
                    # return all_nones[:i] + ('shards',) + all_nones[i+1:]
            raise ValueError(f"Could not shard parameter of shape {shape}")
        train_state_sharding = jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, PartitionSpec(*shard_parameter(spec))), 
            flax.linen.unbox(train_state_shape))

    # Shards a data along the first axis.
    # For single-host, this puts the data on the appropriate device.
    # For multi-host, call this with different data on each host. It will make a global array
    #     representing the data on all hosts, but only part will be addressable on this host.
    def shard_data(batch):
        def _shard_data(x):
            if jax.local_device_count() == jax.device_count():
                return jax.device_put(x, data_sharding)
            else:
                # Increases the first dimension by num_hosts. X is no longer fully addressable.
                x_shape = (x.shape[0] * num_hosts, *x.shape[1:])
                x = np.split(x, len(mesh.local_devices), axis = 0) # per device data, but on host
                x = jax.device_put(x, mesh.local_devices) # per device data, now on device
                return jax.make_array_from_single_device_arrays(x_shape, data_sharding, x)
        return jax.tree_util.tree_map(_shard_data, batch)
        #return multihost_utils.host_local_array_to_global_array(
        #    batch, mesh, PartitionSpec("devices")
        #)

    # Collect a multi-host array onto the local device.
    def global_to_local(x):
        return jax.experimental.multihost_utils.global_array_to_host_local_array(x, mesh, PartitionSpec('devices'))
    
    # The first three are 'Sharding' objects which are pytrees.
    # The last two are helper functions for moving data between devices.
    return data_sharding, train_state_sharding, no_shard, shard_data, global_to_local


def gather_train_state(train_state):
    # Function to all-gather sharded arrays
    def allgather_param(param):
        if isinstance(param, jax.Array):
            # Gather sharded array across all processes
            return process_allgather(param)
        else:
            return param
    # Apply the all-gather function to all parameters in the train state
    gathered_state = jax.tree_util.tree_map(allgather_param, train_state)
    # Move the gathered parameters to host memory
    host_state = jax.device_get(gathered_state)
    return host_state
