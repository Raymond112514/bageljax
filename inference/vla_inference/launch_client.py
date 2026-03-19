import asyncio
import numpy as np
import websockets
from bageljax.utils import msgpack_numpy
from data_sampler import TrajectoryDataSampler
import matplotlib.pyplot as plt
from media_saver import MediaSaver
import imageio

NUM_BUCKETS = 64

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

    sampler = TrajectoryDataSampler(
        data_paths=[
            "gs://raymond-us-west1/droid/failure/failure-00000.tfrecord",
        ],
        seed=np.random.randint(0, 1000000),
    )
    traj = sampler.sample_one_trajectory()
    language_instruction = traj.as_dict()['language_instruction'][0].decode('utf-8').lower()
    file_index = traj.as_dict()['tfrecord_shard_id']
    print_green(f"File index: {file_index}")
    media_saver = MediaSaver(save_dir=f"media/{file_index}_{language_instruction}")
    ep_len = traj.as_dict()['wrist_image'].shape[0]
    print(f"Episode length: {ep_len}")
    all_values = []
    num_devices = 4
    max_bound = (ep_len // num_devices) * num_devices
    all_logits = []
    for t in range(0, max_bound, num_devices):
        print(f"Processing time step {t}")
        wrist_image = traj.as_dict()['wrist_image'][t:t+num_devices]
        shoulder_image = traj.as_dict()['shoulder_image'][t:t+num_devices]
        instruction = [
            x for x in traj.as_dict()['language_instruction'][t : t + num_devices]
        ]
        image = np.concatenate([wrist_image, shoulder_image], axis=1)
        obs = {
            "shoulder_image": shoulder_image,
            "wrist_image": wrist_image,
            "language_instruction": instruction,
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