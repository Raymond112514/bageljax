import os
import numpy as np

from bageljax.data.dataset import Dataset

data_paths = [["/home/pranav/bageljax/tfrecords/success-00007.tfrecord"]]
dataset = Dataset(
    data_paths,
    0,
    action_proprio_metadata={"mean": np.zeros((8,), dtype=np.float32), "std": np.ones((8,), dtype=np.float32)},
    batch_size=2,
    train=False,
)

iterator = dataset.iterator()

batch = next(iterator)
print(type(batch))
for key in batch.keys():
    if "language" not in key:
        print(key, batch[key].shape, batch[key].dtype)
    else:
        print(key, batch[key][0].decode("utf-8"))
        print(key, batch[key][1].decode("utf-8"))