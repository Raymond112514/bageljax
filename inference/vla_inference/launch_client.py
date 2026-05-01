import asyncio
import numpy as np
import websockets
from bageljax.utils import msgpack_numpy
from data_sampler import TrajectoryDataSampler
import matplotlib.pyplot as plt
from media_saver import MediaSaver

NUM_BUCKETS = 64
ACTION_CHUNK_SIZE = 30  # must match training / inference_server INFERENCE_CONFIG
# Query stride for value labeling/inference.
QUERY_STRIDE = ACTION_CHUNK_SIZE
NUM_TRAJ_PER_TFRECORD = 20
MAX_EXAMPLE_SCAN_PER_TFRECORD = 2000

INFERENCE_DATA_PATHS = [
    "gs://raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00040.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00041.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00042.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00043.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00044.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00045.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00046.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00010.tfrecord",
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
            print(f"Pred values shape: {pred_values.shape}")
            print("-" * 50)
            media_saver.save_value_plot(
                queried_images,
                pred_values,
                filename="value_function.mp4",
                title=language_instruction,
                stride=QUERY_STRIDE,
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