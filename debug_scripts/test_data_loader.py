import os
import numpy as np

from bageljax.data.dataset import Dataset

data_paths = [["/home/pranav/bageljax/tfrecords/success-00007.tfrecord"]]
dataset = Dataset(
    data_paths,
    0,
    batch_size=1,
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