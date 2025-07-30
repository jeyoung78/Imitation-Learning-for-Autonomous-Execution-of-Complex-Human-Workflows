import sys, pathlib
# make the repo root importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
import random
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import tqdm
import csv

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

# ─────────────── Static configuration ───────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_DIR = "./data/button"
VQVAE_CHECKPOINT = "./checkpoints_pybullet/buttons/more_more_data/vqvae/vqvae_epoch200.pt"
SAVE_DIR = "./checkpoints_pybullet/buttons/more_data/transformer"
os.makedirs(SAVE_DIR, exist_ok=True)

# VQ-VAE parameters (must match pretraining)
ACTION_WINDOW_SIZE = 1           # single-step coding
ACT_DIM            = 10          # ← match your data_act.npy’s last dim :contentReference[oaicite:0]{index=0}
# BehaviorTransformer context sizes
N_OBS_WINDOW       = 1          # e.g. condition on the last 10 observations
N_ACT_WINDOW       = 1          # predict 1 step ahead
OBS_DIM            = 31          # ← match your data_obs.npy’s last dim :contentReference[oaicite:1]{index=1}

# Transformer & training hyperparameters
TRANSFORMER_LR = 1e-4
BATCH_SIZE     = 4096             # adjust to your dataset size
EPOCHS         = 2000
GRAD_CLIP      = 1.0

# VQ-BeT model hyperparameters (you can leave these)
N_LATENT_DIMS  = 512
# VQVAE_N_EMBED  = 16
VQVAE_N_EMBED  = 32
VQVAE_GROUPS   = 2
# smaller model size
GPT_EMB_DIM    = 128
GPT_N_LAYER    = 6
GPT_N_HEAD     = 8
# GPT_EMB_DIM    = 256
# GPT_N_LAYER    = 12
# GPT_N_HEAD     = 8
GPT_BLOCK_SIZE = N_OBS_WINDOW

def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class UR3SequenceDataset(Dataset):
    """
    Sliding windows of obs+act for UR3, with mask aligned to sequence length.
    """
    def __init__(self, root_dir: str,
                 n_obs_window: int,
                 n_act_window: int):
        # 1) Load
        obs_raw = np.load(Path(root_dir) / "data_obs.npy")
        act_raw = np.load(Path(root_dir) / "data_act.npy")
        msk_raw = np.load(Path(root_dir) / "data_msk.npy").astype(bool)

        # 2) Ensure episode-first shapes
        if obs_raw.ndim == 2:
            obs_raw = obs_raw[np.newaxis, ...]
            act_raw = act_raw[np.newaxis, ...]
            msk_raw = msk_raw[np.newaxis, ...]
        E, T, O = obs_raw.shape
        _, T2, A  = act_raw.shape
        assert T == T2, f"obs T={T} vs act T={T2}"

        # 3) Squeeze any trailing mask dim, then broadcast/repeat
        if msk_raw.ndim == 3 and msk_raw.shape[-1] == 1:
            msk_raw = msk_raw.squeeze(-1)
        if msk_raw.ndim == 1:
            msk_raw = np.broadcast_to(msk_raw, (E, msk_raw.shape[0]))
        if msk_raw.ndim == 2 and msk_raw.shape[0] == 1:
            msk_raw = np.repeat(msk_raw, E, axis=0)

        # 4) Trim or pad mask to length T
        _, T_mask = msk_raw.shape
        if T_mask > T:
            msk_raw = msk_raw[:, :T]
        elif T_mask < T:
            # pad False for missing timesteps
            pad = np.zeros((E, T - T_mask), dtype=bool)
            msk_raw = np.concatenate([msk_raw, pad], axis=1)

        # final check
        if msk_raw.shape != (E, T):
            raise RuntimeError(f"Mask shape {msk_raw.shape} ≠ (E={E}, T={T})")

        # 5) Slide windows
        self.samples = []
        total_win = n_obs_window + n_act_window
        for e in range(E):
            obs_e  = obs_raw[e]   # (T, O)
            act_e  = act_raw[e]   # (T, A)
            mask_e = msk_raw[e]   # (T,)
            for t in range(T - total_win + 1):
                i_obs = slice(t, t + n_obs_window)
                i_act = slice(t + n_obs_window, t + total_win)
                if mask_e[i_obs].all() and mask_e[i_act].all():
                    self.samples.append((
                        obs_e[i_obs].astype(np.float32),
                        act_e[i_act].astype(np.float32)
                    ))

        if not self.samples:
            raise RuntimeError(f"No valid windows in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, a = self.samples[idx]
        return (
            torch.from_numpy(o).float().clone(),
            torch.from_numpy(a).float().clone()
        )


# ─────────────── Model setup ───────────────
from vqvae.vqvae import VqVae
from vq_behavior_transformer.gpt import GPT, GPTConfig
from vq_behavior_transformer.bet import BehaviorTransformer

# Load pre-trained VQ-VAE
vqvae = VqVae(
    input_dim_h=ACTION_WINDOW_SIZE,
    input_dim_w=ACT_DIM,
    n_latent_dims=N_LATENT_DIMS,
    vqvae_n_embed=VQVAE_N_EMBED,
    vqvae_groups=VQVAE_GROUPS,
    eval=True,
    device=DEVICE
)
vqvae.load_state_dict(torch.load(VQVAE_CHECKPOINT, map_location=DEVICE))

# Build GPT
gpt_cfg = GPTConfig(
    input_dim=OBS_DIM,
    output_dim=VQVAE_N_EMBED * VQVAE_GROUPS,
    block_size=GPT_BLOCK_SIZE,
    n_layer=GPT_N_LAYER,
    n_head=GPT_N_HEAD,
    n_embd=GPT_EMB_DIM,
    dropout=0.1
)
gpt = GPT(gpt_cfg).to(DEVICE)

# Build BehaviorTransformer
agent = BehaviorTransformer(
    obs_dim=OBS_DIM,
    act_dim=ACT_DIM,
    goal_dim=0,
    gpt_model=gpt,
    vqvae_model=vqvae,
    offset_loss_multiplier=0.5,
    obs_window_size=N_OBS_WINDOW,
    act_window_size=N_ACT_WINDOW
).to(DEVICE)


# Optimizer
optim = AdamW(agent.parameters(), lr=TRANSFORMER_LR)
loss_arr = []
# ─────────────── DataLoader & Training Loop ───────────────
if __name__ == '__main__':
    train_ds = UR3SequenceDataset(DATA_DIR, N_OBS_WINDOW, N_ACT_WINDOW)
    print("Total training windows:", len(train_ds))
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
    with open(f"data/button/more_data_loss_convergence.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames =  ["Epoch", "Loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, EPOCHS+1):
            total_loss = 0.0
            for obs_seq, act_seq in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
                obs_seq = obs_seq.to(DEVICE)
                act_seq = act_seq.to(DEVICE)
                
                # agent returns (predictions, total_loss, metrics_dict)
                _, loss, metrics = agent(obs_seq, None, act_seq)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                optim.step()
                total_loss += loss.item()

            avg = total_loss / len(train_loader)
            writer.writerow({"Epoch": epoch, "Loss": avg})
            # loss_arr.append(avg)
            print(f"[Epoch {epoch:03d}] Avg Loss: {avg:.4f}  CodeAcc: {metrics.get('code_acc',0):.3f}")
            if epoch % 20 == 0 or epoch == EPOCHS - 1:
                path = os.path.join(SAVE_DIR, f"agent_epoch{epoch:03d}.pt")
                torch.save(agent.state_dict(), path)
                print(f"--> Saved agent checkpoint: {path}")

            if avg < 0.02:
                break
    
        
        # print(loss_arr)
        torch.save(agent.state_dict(), os.path.join(SAVE_DIR, "agent_final.pt"))
        
        print("Training complete. Model saved.")
