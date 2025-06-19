import os
from PIL import Image
import numpy as np

# --- Config ---
N = 5                # number of samples
seq_len = 20         # sequence length
action_dim = 7      # action dimension

# Use a raw string for the Windows path
out_dir = r"C:\Users\jupar\imitationLearning\data"
img_dir1 = os.path.join(out_dir, 'images1')
img_dir2 = os.path.join(out_dir, 'images2')

# --- Create folders ---
os.makedirs(img_dir1, exist_ok=True)
os.makedirs(img_dir2, exist_ok=True)

# --- Generate & save coords.npy ---
coords = np.random.randn(N, seq_len, action_dim).astype(np.float32)
np.save(os.path.join(out_dir, 'coords.npy'), coords)

# Save one history sequence for inference
np.save(os.path.join(out_dir, 'init_coords.npy'), coords[0])

# --- Generate & save random PNGs ---
for i in range(N):
    # random 64×64 RGB images
    arr1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    arr2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr1).save(os.path.join(img_dir1, f'view1_{i:03d}.png'))
    Image.fromarray(arr2).save(os.path.join(img_dir2, f'view2_{i:03d}.png'))

print("Created:")
print(" ", img_dir1, "→", sorted(os.listdir(img_dir1)))
print(" ", img_dir2, "→", sorted(os.listdir(img_dir2)))
print(" ", os.path.join(out_dir, 'coords.npy'), "shape =", np.load(os.path.join(out_dir, 'coords.npy')).shape)
print(" ", os.path.join(out_dir, 'init_coords.npy'), "shape =", np.load(os.path.join(out_dir, 'init_coords.npy')).shape)
