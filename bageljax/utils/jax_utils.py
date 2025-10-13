import logging
import os
from typing import Any, Optional, Sequence
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from jax.experimental import multihost_utils, mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax import core as jcore
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

_CURRENT_MESH: Optional[Mesh] = None
_SHARDING_CONSTRAINTS_ON: bool = True

def get_current_mesh() -> Mesh:
    assert _CURRENT_MESH is not None, "Mesh not initialized; call create_sharding first."
    return _CURRENT_MESH

def enforce_sharding_constraints(enforce: bool):
    global _SHARDING_CONSTRAINTS_ON
    _SHARDING_CONSTRAINTS_ON = enforce

def is_sharding_active():
    return _SHARDING_CONSTRAINTS_ON

def add_batch_sharding_constraint(x, *, axis_name='devices', where=''):
    """
    Enforce that `x` is sharded on its leading (batch) axis across `axis_name`.
    Wrap the constraint in a named_call so compiler errors include `where`.
    Returns `x` (or constrained `x`).
    """
    if not _SHARDING_CONSTRAINTS_ON:
        return x
    if not isinstance(x, (jax.Array, jcore.Tracer)):
        return x

    pspec = PartitionSpec(axis_name, *([None] * (x.ndim - 1)))

    def _constrain(v):
        return lax.with_sharding_constraint(v, pspec)

    # Name this call so compilation errors include the `where` tag.
    name = f"add_batch_sharding_constraint[{where}]" if where else "add_batch_sharding_constraint"
    return jax.named_call(_constrain, name=name)(x)

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
    mesh = Mesh(devices=device_mesh, axis_names=('devices',))
    jax.set_mesh(mesh) # a mesh in-context is needed for specifying sharding constraints later
    global _CURRENT_MESH
    _CURRENT_MESH = mesh
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
            # Leave non-array-ish leaves alone (e.g., strings)
            if not hasattr(x, "shape"):
                return x

            # If it's already a JAX Array (e.g., from a previous stage), don't re-shard.
            if isinstance(x, jax.Array):
                return x

            # Ensure NumPy for clean host-side splitting
            x = np.asarray(x)

            if jax.local_device_count() == jax.device_count():
                # Single-host: all devices are local; device_put to NamedSharding is fine
                return jax.device_put(x, data_sharding)
            else:
                # Multi-host: create a *global* array from per-local-device arrays
                local_dev_count = jax.local_device_count()
                if x.shape[0] % local_dev_count != 0:
                    raise ValueError(
                        f"shard_data: leading dim {x.shape[0]} not divisible by "
                        f"local_device_count {local_dev_count}"
                    )

                # Split host batch into per-local-device chunks
                parts = np.split(x, local_dev_count, axis=0)  # list of np arrays

                # Place each chunk on a local device
                per_device_arrays = [
                    jax.device_put(p, d) for p, d in zip(parts, jax.local_devices())
                ]

                # Assemble a global array with the NamedSharding defined above
                global_shape = (x.shape[0] * num_hosts, *x.shape[1:])
                return jax.make_array_from_single_device_arrays(
                    global_shape, data_sharding, per_device_arrays
                )

        return jax.tree_util.tree_map(_shard_data, batch)

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
