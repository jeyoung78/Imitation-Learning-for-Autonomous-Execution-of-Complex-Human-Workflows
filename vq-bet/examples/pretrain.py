import sys, pathlib
# make the repo root importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

# ─────────────────── Config ───────────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)
DATA_PATH           = "./data"
SAVE_PATH           = "./checkpoints_pybullet/vqvae"
BATCH_SIZE          = 2048
EPOCHS              = 300
CHECKPOINT_INTERVAL = 10

ACTION_WINDOW_SIZE = 1
ACT_DIM            = 14
N_LATENT_DIMS      = 512
VQVAE_N_EMBED      = 16
VQVAE_GROUPS       = 2

# ────────────────── Helpers ───────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class UR3Dataset(Dataset):
    def __init__(self, data_dir: str):
        acts = np.load(Path(data_dir) / "data_act.npy")    # (N, T, D)
        _, _, D = acts.shape
        # ACTION_WINDOW_SIZE = 1
        ACT_DIM = D
        mask = np.load(Path(data_dir) / "data_msk.npy")    # (N, T, 1) or (N, T)
        mask = mask.squeeze(-1) if mask.ndim == 3 else mask
        mask = (mask > 0).astype(bool)
        N, T, _      = acts.shape
        #print(D)
        windows   = []
        for i in range(N):
            # print(N)
            traj   = acts[i]    # (T, D)
            # print(traj)
            for t in range(T - ACTION_WINDOW_SIZE + 1):
                windows.append(traj[t : t+ACTION_WINDOW_SIZE].astype(np.float32))
                # print(traj[t : t+ACTION_WINDOW_SIZE])
                # print(T)
        if not windows:
            raise RuntimeError(f"No valid windows in {data_dir}")
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])  # (W, D)

from vqvae.vqvae import VqVae

def train_loop(
    vqvae,
    loader,
    device,
    epochs: int,
    checkpoint_interval: int,
    save_path: str
):
    """
    Runs VQ-VAE training for `epochs` epochs on `loader`, 
    logging encoder/VQ/recon losses and checkpointing.
    """
    for epoch in range(1, epochs + 1):
        sum_enc, sum_vq, sum_recon = 0.0, 0.0, 0.0
        # single progress bar for entire epoch
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        first = True
        for batch in pbar:
            # print(batch)
            batch = batch.to(device)
            enc, vq, codes, recon, dec_out = vqvae.vqvae_update(batch)
            if first:
                print(f"Input action: {batch[0]}")
                print(f"Decoder out:  {dec_out[0]}")
                print(f"loss: {recon}")
                first = False

            # accumulate
            sum_enc   += enc.item()
            sum_vq    += vq.item()
            sum_recon += recon

        # compute averages
        iters = len(loader)
        if iters > 0:   
            avg_enc   = sum_enc   / iters
            avg_vq    = sum_vq    / iters
            avg_recon = sum_recon / iters

        # checkpoint
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            fname = f"vqvae_epoch{epoch:03d}.pt"
            path  = os.path.join(save_path, fname)
            torch.save(vqvae.state_dict(), path)
            print(f"→ checkpoint saved: {path}")


def main():
    seed_everything(SEED)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # model
    vqvae = VqVae(
        input_dim_h=ACTION_WINDOW_SIZE,
        input_dim_w=ACT_DIM,
        n_latent_dims=N_LATENT_DIMS,
        vqvae_n_embed=VQVAE_N_EMBED,
        vqvae_groups=VQVAE_GROUPS,
        eval=False,
        device=DEVICE
    )

    # data
    dataset = UR3Dataset(DATA_PATH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True, num_workers=4)

    # training
    train_loop(vqvae, loader, DEVICE, EPOCHS, CHECKPOINT_INTERVAL, SAVE_PATH)

    # final save
    final_ckpt = os.path.join(SAVE_PATH, f"vqvae_epoch{EPOCHS:03d}.pt")
    torch.save(vqvae.state_dict(), final_ckpt)
    print(f"Training complete. Final model: {final_ckpt}")

if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
