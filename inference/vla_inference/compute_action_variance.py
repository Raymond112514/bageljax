import asyncio
import numpy as np
import websockets
from bageljax.utils import msgpack_numpy
from data_sampler import TrajectoryDataSampler, TrajectoryObservation

NUM_BUCKETS = 64
ACTION_CHUNK_SIZE = 30
# Stride over timesteps for which observation (shoulder, instruction) to query.
OBS_STRIDE = 30
# Number of random action chunks per observation; also value-function batch size.
K_SAMPLES = 32
NUM_TRAJ_PER_TFRECORD = 20
MAX_EXAMPLE_SCAN_PER_TFRECORD = 2000
SERVER_MIN_BATCH = 4  # must be divisible by server device count

TRAIN_DATA_PATHS = [
    "gs://raymond-us-west1/droid/roboarena/roboarena-00000.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00005.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00010.tfrecord",
]

TEST_DATA_PATHS = [
    "gs://raymond-us-west1/droid/roboarena/roboarena-00041.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00043.tfrecord",
    "gs://raymond-us-west1/droid/roboarena/roboarena-00045.tfrecord",
]

TRAIN_REWARD_NPY = "train_reward.npy"
TEST_REWARD_NPY = "test_reward.npy"


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_value(logits, num_buckets=64):
    logits = logits[:, :num_buckets].astype(np.float32)
    probs = softmax(logits, axis=-1)
    k = logits.shape[-1]
    bucket_ids = np.arange(k)
    value = np.sum(probs * bucket_ids, axis=-1)
    value = NUM_BUCKETS - value - 1
    return value


async def infer_values(ws, packer, shoulder_image, instruction, action_chunks):
    """Run inference for multiple candidate action chunks; returns scalar value per row."""
    n = action_chunks.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    pad = (-n) % SERVER_MIN_BATCH
    if pad > 0:
        shoulder_image = np.concatenate(
            [shoulder_image, shoulder_image[:1].repeat(pad, axis=0)], axis=0
        )
        action_chunks = np.concatenate(
            [action_chunks, action_chunks[:1].repeat(pad, axis=0)], axis=0
        )
        instruction = list(instruction) + [instruction[0]] * pad

    obs = {
        "shoulder_image": shoulder_image,
        "language_instruction": instruction,
        "action/joint_velocity_chunk": action_chunks.astype(np.float32),
    }
    await ws.send(packer.pack(obs))
    logits = msgpack_numpy.unpackb(await ws.recv())
    logits = np.asarray(logits)
    values = compute_value(logits)
    return values[:n]


async def rewards_for_trajectory(
    traj: TrajectoryObservation,
    ws,
    packer,
    *,
    rng: np.random.Generator,
    obs_stride: int = OBS_STRIDE,
    k_samples: int = K_SAMPLES,
    action_chunk_size: int = ACTION_CHUNK_SIZE,
) -> np.ndarray:
    """
    For trajectory `traj`, at observation indices 0, obs_stride, 2*obs_stride, ...
    sample `k_samples` action chunks i.i.d. N(0, I) of shape (action_chunk_size, 8),
    run the value head, and return rewards of shape (num_strided_states, k_samples).
    """
    d = traj.as_dict()
    ep_len = int(d["shoulder_image"].shape[0])
    state_indices = np.arange(0, ep_len, obs_stride, dtype=np.int32)
    num_states = int(state_indices.shape[0])
    if num_states == 0:
        return np.zeros((0, k_samples), dtype=np.float32)

    out = np.zeros((num_states, k_samples), dtype=np.float32)
    for row, s in enumerate(state_indices):
        action_chunks = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(k_samples, action_chunk_size, 8),
        ).astype(np.float32)
        shoulder_batch = np.repeat(
            d["shoulder_image"][s : s + 1],
            repeats=k_samples,
            axis=0,
        )
        instruction_batch = [d["language_instruction"][s]] * k_samples
        values = await infer_values(
            ws=ws,
            packer=packer,
            shoulder_image=shoulder_batch,
            instruction=instruction_batch,
            action_chunks=action_chunks,
        )
        out[row, :] = values
    return out


async def collect_reward_matrix(
    data_paths: list[str],
    uri: str,
    *,
    rng: np.random.Generator,
    num_traj_per_tfrecord: int = NUM_TRAJ_PER_TFRECORD,
) -> np.ndarray:
    """Concatenate (num_states, K) blocks from trajectories across all paths into (N, K)."""
    blocks: list[np.ndarray] = []
    packer = msgpack_numpy.Packer()

    def print_green(text: str) -> None:
        print(f"\033[92m{text}\033[0m")

    async with websockets.connect(uri, max_size=None, compression=None) as ws:
        _cfg = msgpack_numpy.unpackb(await ws.recv())

        for data_path in data_paths:
            sampler = TrajectoryDataSampler(data_paths=[data_path])
            print_green(
                f"Collecting up to {num_traj_per_tfrecord} trajectories from {data_path}"
            )
            num_collected = 0
            example_index = 0
            while (
                num_collected < num_traj_per_tfrecord
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

                mat = await rewards_for_trajectory(
                    traj, ws, packer, rng=rng
                )
                if mat.size > 0:
                    blocks.append(mat)
                    print_green(
                        f"  traj {num_collected}: rows={mat.shape[0]}, K={mat.shape[1]}"
                    )

            if num_collected < num_traj_per_tfrecord:
                print(
                    f"Only collected {num_collected}/{num_traj_per_tfrecord} from "
                    f"{data_path} after scanning {example_index} examples"
                )

    if not blocks:
        raise ValueError("No reward rows collected; check paths and server.")
    return np.concatenate(blocks, axis=0)


async def main():
    uri = "ws://localhost:8000"
    rng = np.random.default_rng(0)

    train_mat = await collect_reward_matrix(
        TRAIN_DATA_PATHS, uri, rng=rng
    )
    np.save(TRAIN_REWARD_NPY, train_mat)
    print(f"Saved {train_mat.shape} -> {TRAIN_REWARD_NPY}")

    rng = np.random.default_rng(1)
    test_mat = await collect_reward_matrix(
        TEST_DATA_PATHS, uri, rng=rng
    )
    np.save(TEST_REWARD_NPY, test_mat)
    print(f"Saved {test_mat.shape} -> {TEST_REWARD_NPY}")


if __name__ == "__main__":
    asyncio.run(main())
