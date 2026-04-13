import asyncio
import numpy as np
import websockets
from bageljax.utils import msgpack_numpy
from data_sampler import TrajectoryDataSampler
import matplotlib.pyplot as plt
from media_saver import MediaSaver
import imageio

NUM_BUCKETS = 64
ACTION_CHUNK_SIZE = 30  # must match training / inference_server INFERENCE_CONFIG

INFERENCE_DATA_PATHS = [
    "gs://raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00005.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00010.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00015.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00020.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00025.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00030.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00035.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00040.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00045.tfrecord",
]

# INFERENCE_DATA_PATHS = [
#     "gs://raymond-us-west1/droid/success/success-00000.tfrecord",
#     "gs://raymond-us-west1/droid/success/success-00005.tfrecord",
#     "gs://raymond-us-west1/droid/success/success-00010.tfrecord",
#     "gs://raymond-us-west1/droid/success/success-00015.tfrecord",
#     "gs://raymond-us-west1/droid/success/success-00020.tfrecord",
# ]

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
        sampler = TrajectoryDataSampler(
            data_paths=[data_path]
        )
        traj = sampler.sample_one_trajectory()
        language_instruction = traj.as_dict()['language_instruction'][0].decode('utf-8').lower()
        file_index = traj.as_dict()['tfrecord_shard_id']
        print_green(f"File index: {file_index}")
        media_saver = MediaSaver(save_dir=f"media/{file_index}_{language_instruction}")
        d = traj.as_dict()
        ep_len = d["wrist_image"].shape[0]
        full_action_8d = d["action/joint_velocity"]  # (T, 8)
        print(f"Episode length: {ep_len}")
        all_values = []
        num_devices = 4
        K = ACTION_CHUNK_SIZE
        # Need t + num_devices - 1 + K <= ep_len  =>  t <= ep_len - K - num_devices + 1
        max_t = ep_len - K - num_devices + 1
        if max_t < 0:
            print(f"Skip episode: ep_len {ep_len} too short for K={K} and batch {num_devices}")
            continue
        all_logits = []
        for t in range(0, max_t + 1, num_devices):
            print(f"Processing time step {t}")
            wrist_image = d["wrist_image"][t : t + num_devices]
            shoulder_image = d["shoulder_image"][t : t + num_devices]
            action_chunks = np.stack(
                [full_action_8d[t + i : t + i + K] for i in range(num_devices)],
                axis=0,
            )
            print(f"action/joint_velocity_chunk shape: {action_chunks.shape}")
            print(f"shoulder image shape: {shoulder_image.shape}")
            print(f"wrist image shape: {wrist_image.shape}")
            instruction = [
                x for x in d["language_instruction"][t : t + num_devices]
            ]
            obs = {
                "shoulder_image": shoulder_image,
                "wrist_image": wrist_image,
                "language_instruction": instruction,
                "action/joint_velocity_chunk": action_chunks.astype(np.float32),
            }
            async with websockets.connect(uri, max_size=None, compression=None) as ws:
                cfg = msgpack_numpy.unpackb(await ws.recv())
                print("server config:", cfg)

                await ws.send(packer.pack(obs))
                logits = msgpack_numpy.unpackb(await ws.recv())
                logits = np.asarray(logits)
                value = compute_value(logits)
                all_values.extend(list(value))
                all_logits.extend(list(logits))
                
        all_logits = np.asarray(all_logits)

        media_saver.save_value_plot(
            traj.as_dict()['image'],
            all_values,
            filename="value_function.mp4",
            title=language_instruction,
        )
        plot_heatmap(all_logits, path=f"media/{file_index}_{language_instruction}/heatmap.png") 
        
        # Save both camera views
        imageio.mimwrite(f"media/{file_index}_{language_instruction}/shoulder_view.mp4", traj.as_dict()['shoulder_image'])
        imageio.mimwrite(f"media/{file_index}_{language_instruction}/wrist_view.mp4", traj.as_dict()['wrist_image'])
    
asyncio.run(main())