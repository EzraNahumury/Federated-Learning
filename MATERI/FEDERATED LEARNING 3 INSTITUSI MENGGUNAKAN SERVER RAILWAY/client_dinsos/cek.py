import numpy as np

npz = np.load("models/global_model_fedavg.npz", allow_pickle=True)
print(f"Total tensor: {len(npz.files)}")
for i, key in enumerate(npz.files):
    print(f"{i}: {npz[key].shape}")
