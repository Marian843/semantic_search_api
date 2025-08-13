import numpy as np

metadata = np.load("data/metadata.npy", allow_pickle=True)
print(type(metadata), len(metadata))
print(metadata[0])  # first entry
