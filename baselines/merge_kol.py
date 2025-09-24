import numpy as np
import os
from tqdm import tqdm
kol_pth = '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/raw_data/train/kf_2d_re1000_256_40seed.npy'

output_dir = '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/kolmogorov'
os.makedirs(output_dir, exist_ok=True)

kol = np.load(kol_pth)

mean = 0
std = 12  # approx 1% and 99% percentile of data
# Normalize data
kol = (kol - mean) / std

print(kol.shape)
# Get dimensions
B, T, H, W = kol.shape

# For each batch and time step (except last 2 frames)
count = 0
for b in tqdm(range(B), total=B):
    for t in tqdm(range(T-2), total=T-2):
        # Get 3 consecutive frames
        frames = kol[b, t:t+3]  # Shape: (3, H, W)
        
        # Rearrange to (H, W, 3)
        frames = np.transpose(frames, (1, 2, 0))
        
        # Save to file
        output_path = os.path.join(output_dir, f'kol_{count}.npy')
        np.save(output_path, frames)
        
        # Increment counter
        count += 1
        
        if count % 500 == 0:
            print(f"Processed {count} samples")
            print("Min:", frames.min(), "Max:", frames.max())

print(f"Finished processing {count} total samples")
