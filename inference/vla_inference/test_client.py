import asyncio
import numpy as np
import websockets
from bageljax.utils import msgpack_numpy

async def main():
    uri = "ws://localhost:8000"
    packer = msgpack_numpy.Packer()

    shoulder = np.zeros((288, 512, 3), dtype=np.uint8)
    wrist = np.zeros((288, 512, 3), dtype=np.uint8)
    instruction = b"pick up the red block"

    obs = {
        "shoulder_image": shoulder,
        "wrist_image": wrist,
        "language_instruction": instruction,
    }

    async with websockets.connect(uri, max_size=None, compression=None) as ws:
        cfg = msgpack_numpy.unpackb(await ws.recv())
        print("server config:", cfg)

        await ws.send(packer.pack(obs))
        logits = msgpack_numpy.unpackb(await ws.recv())
        print("logits shape:", np.asarray(logits))

asyncio.run(main())
