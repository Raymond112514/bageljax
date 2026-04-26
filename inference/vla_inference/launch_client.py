import asyncio
import numpy as np
import websockets
import tensorflow as tf
from bageljax.utils import msgpack_numpy
from data_sampler import TrajectoryDataSampler
import matplotlib.pyplot as plt
from media_saver import MediaSaver
import imageio

NUM_BUCKETS = 64
ACTION_CHUNK_SIZE = 30  # must match training / inference_server INFERENCE_CONFIG
# Query stride for value labeling/inference.
QUERY_STRIDE = ACTION_CHUNK_SIZE
# RoboArena GT rewards are pre-labeled every 10 timesteps.
GT_REWARD_TIMESTEP_STRIDE = 10
NUM_TRAJ_PER_TFRECORD = 20
MAX_EXAMPLE_SCAN_PER_TFRECORD = 2000
ROBOARENA_LABELED_PREFIX = "gs://raymond-us-west1/droid_labeled/roboarena_renamed"

INFERENCE_DATA_PATHS = [
    "gs://raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00040.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00041.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00042.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00043.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00044.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00045.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00010.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00020.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00030.tfrecord",
]

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Compute the value from the estimated logits, the smaller value, the closer to the goal
# The value is the expected bucket ID under the estimated probability distribution
def compute_value(logits, num_buckets=64):
    logits = logits[:, :num_buckets].astype(np.float32)
    probs = softmax(logits, axis=-1)
    K = logits.shape[-1]
    bucket_ids = np.arange(K)
    value = np.sum(probs * bucket_ids, axis=-1) 
    value = NUM_BUCKETS - value - 1
    return value

def plot_heatmap(logits, path):
    probs = softmax(logits[:, :64], axis=-1)  
    heatmap = probs[:, ::-1].T 
    plt.figure(figsize=(8, 4))
    plt.imshow(
        heatmap,           
        aspect="auto",
        origin="lower",
        cmap="magma",    
    )
    plt.colorbar(label="Probability")
    plt.xlabel("Timestep")
    plt.ylabel("Class")
    plt.title("Distribution over classes (time)")
    plt.savefig(path)

def load_roboarena_ground_truth_buckets(shard_id, example_index, action_chunk_size, num_buckets):
    """Load GT value targets from labeled RoboArena TFRecord and map to bucket space."""
    if shard_id is None:
        return None
    labeled_path = f"{ROBOARENA_LABELED_PREFIX}/roboarena-{int(shard_id):05d}.tfrecord"
    ds = tf.data.TFRecordDataset([labeled_path]).skip(int(example_index)).take(1)
    try:
        serialized_example = next(iter(ds))
    except StopIteration:
        return None

    parsed = tf.io.parse_single_example(
        serialized_example,
        {"rewards": tf.io.VarLenFeature(tf.float32)},
    )
    rewards = tf.reshape(tf.sparse.to_dense(tf.sparse.reorder(parsed["rewards"])), [-1]).numpy()
    if rewards.shape[0] == 0:
        return None

    # RoboArena labeled rewards are already aligned to action-chunked samples.
    gt_buckets = np.clip(rewards * float(num_buckets), 0.0, float(num_buckets - 1))
    return gt_buckets.astype(np.float32)

async def main():
    uri = "ws://localhost:8000"
    packer = msgpack_numpy.Packer()
    
    def print_green(text):
        print(f"\033[92m{text}\033[0m")
        
    for data_path in INFERENCE_DATA_PATHS:
        sampler = TrajectoryDataSampler(data_paths=[data_path])
        print_green(f"Sampling up to {NUM_TRAJ_PER_TFRECORD} trajectories from {data_path}")
        num_collected = 0
        example_index = 0

        while (
            num_collected < NUM_TRAJ_PER_TFRECORD
            and example_index < MAX_EXAMPLE_SCAN_PER_TFRECORD
        ):
            try:
                traj = sampler.sample_one_trajectory(
                    example_index=example_index,
                    max_attempts=1,
                )
            except ValueError:
                example_index += 1
                continue
            example_index += 1
            num_collected += 1

            language_instruction = traj.as_dict()["language_instruction"][0].decode(
                "utf-8"
            ).lower()
            file_index = traj.as_dict()["tfrecord_shard_id"]
            tfrecord_example_index = traj.as_dict()["tfrecord_example_index"]
            print_green(
                f"File index: {file_index}, trajectory {num_collected}/{NUM_TRAJ_PER_TFRECORD}"
            )
            media_saver = MediaSaver(
                save_dir=f"media/{file_index}_{num_collected:03d}_{language_instruction}"
            )
            d = traj.as_dict()
            ep_len = d["shoulder_image"].shape[0]
            full_action_8d = d["action/joint_velocity"]  # (T, 8)
            all_values = []
            num_devices = 4
            K = ACTION_CHUNK_SIZE
            max_start = ep_len - K
            if max_start < 0:
                print(
                    f"Skip episode: ep_len {ep_len} too short for K={K} and batch {num_devices}"
                )
                continue
            start_indices = list(range(0, max_start + 1, QUERY_STRIDE))
            if len(start_indices) < num_devices:
                print(
                    f"Skip episode: only {len(start_indices)} chunk starts for "
                    f"K={K}, stride={QUERY_STRIDE}, batch={num_devices}"
                )
                continue
            print(
                f"Chunk starts: {len(start_indices)} "
                f"(stride={QUERY_STRIDE}, first={start_indices[0]}, last={start_indices[-1]})"
            )
            all_logits = []
            queried_start_indices = []
            async with websockets.connect(uri, max_size=None, compression=None) as ws:
                cfg = msgpack_numpy.unpackb(await ws.recv())
                print("server config:", cfg)
                for i in range(0, len(start_indices) - num_devices + 1, num_devices):
                    starts = start_indices[i : i + num_devices]
                    print(
                        f"Processing chunk starts {starts[0]}..{starts[-1]} "
                        f"(batch {i // num_devices + 1})"
                    )
                    shoulder_image = d["shoulder_image"][starts]
                    action_chunks = np.stack(
                        [full_action_8d[s : s + K] for s in starts],
                        axis=0,
                    )
                    instruction = [d["language_instruction"][s] for s in starts]
                    # Shoulder-only vision matches training (Dataset._stack_and_reshape_images).
                    obs = {
                        "shoulder_image": shoulder_image,
                        "language_instruction": instruction,
                        "action/joint_velocity_chunk": action_chunks.astype(np.float32),
                    }
                    await ws.send(packer.pack(obs))
                    logits = msgpack_numpy.unpackb(await ws.recv())
                    logits = np.asarray(logits)
                    value = compute_value(logits)
                    all_values.extend(list(value))
                    all_logits.extend(list(logits))
                    queried_start_indices.extend(starts)

            all_logits = np.asarray(all_logits)
            pred_values = np.asarray(all_values, dtype=np.float32)
            queried_images = d["image"][queried_start_indices]
            gt_values = load_roboarena_ground_truth_buckets(
                shard_id=file_index,
                example_index=tfrecord_example_index,
                action_chunk_size=K,
                num_buckets=NUM_BUCKETS,
            )
            print(f"Pred values shape: {pred_values.shape}")
            print(f"GT values shape: {gt_values.shape}")
            print("-"*50)
            queried_gt_values = None
            if gt_values is not None:
                qidx = np.asarray(queried_start_indices, dtype=np.int32)
                max_q = int(np.max(qidx)) if qidx.size > 0 else -1
                if gt_values.shape[0] > max_q:
                    # GT indexed on timestep domain.
                    queried_gt_values = gt_values[qidx]
                else:
                    # GT indexed on pre-labeled domain (one label per 10 timesteps).
                    chunk_idx = qidx // int(GT_REWARD_TIMESTEP_STRIDE)
                    max_c = int(np.max(chunk_idx)) if chunk_idx.size > 0 else -1
                    if gt_values.shape[0] > max_c:
                        queried_gt_values = gt_values[chunk_idx]

            media_saver.save_value_plot(
                queried_images,
                pred_values,
                filename="value_function.mp4",
                title=language_instruction,
                stride=QUERY_STRIDE,
                ground_truth_values=queried_gt_values,
            )
            plot_heatmap(
                all_logits,
                path=f"media/{file_index}_{num_collected:03d}_{language_instruction}/heatmap.png",
            )

        if num_collected < NUM_TRAJ_PER_TFRECORD:
            print(
                f"Collected {num_collected}/{NUM_TRAJ_PER_TFRECORD} trajectories from {data_path} "
                f"after scanning {example_index} examples"
            )
    
asyncio.run(main())