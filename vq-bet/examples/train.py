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

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

# ─────────────── Static configuration ───────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_DIR = r"C:\Users\jupar\Downloads\vqbet_datasets_for_release\vqbet_datasets_for_release\ur3"
VQVAE_CHECKPOINT = r"./checkpoints/ur3_vqvae/vqvae_epoch150.pt"
SAVE_DIR = "./checkpoints/ur3_transformer"
os.makedirs(SAVE_DIR, exist_ok=True)

# VQ-VAE parameters (must match pretraining)
ACTION_WINDOW_SIZE = 1    # single-step VQ coding
ACT_DIM = 2              # UR3 7-DoF
# BehaviorTransformer context sizes\N_OBS_WINDOW = 10     # number of past obs steps
N_ACT_WINDOW = 1        # number of past action steps
OBS_DIM = 6          # will infer from data_obs

# Transformer & training hyperparameters
TRANSFORMER_LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
GRAD_CLIP = 1.0

# VQ-BeT model hyperparameters
N_LATENT_DIMS = 512
VQVAE_N_EMBED = 16
VQVAE_GROUPS = 2
GPT_EMB_DIM = 128
GPT_N_LAYER = 6
GPT_N_HEAD = 8
N_OBS_WINDOW = 1
GPT_BLOCK_SIZE = N_OBS_WINDOW 

def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ─────────────── Dataset ───────────────
class UR3SequenceDataset(Dataset):
    """
    Returns sliding windows of observations and actions for UR3.
    Handles per-episode or flat formats for data_obs.npy, data_act.npy, data_msk.npy.
    """
    def __init__(self, root_dir:str):
        obs_raw = np.load(Path(root_dir)/"data_obs.npy")
        act_raw = np.load(Path(root_dir)/"data_act.npy")
        msk_raw = np.load(Path(root_dir)/"data_msk.npy").astype(bool)

        # Broadcast single trajectory to episode-first format
        if obs_raw.ndim == 2:
            obs_raw = obs_raw[np.newaxis, ...]
            act_raw = act_raw[np.newaxis, ...]
            msk_raw = msk_raw[np.newaxis, ...]
        E, T, O = obs_raw.shape
        _, T_act, A = act_raw.shape
        assert T == T_act, "obs and act sequence lengths must match"

        global OBS_DIM
        OBS_DIM = O
        self.samples = []

        for e in range(E):
            obs_seq_e = obs_raw[e]
            act_seq_e = act_raw[e]
            mask_e    = msk_raw[e]
            for t in range(T - (max(N_OBS_WINDOW, N_ACT_WINDOW)) + 1):
                idx_obs = slice(t, t + N_OBS_WINDOW)
                idx_act = slice(t + N_OBS_WINDOW, t + N_OBS_WINDOW + N_ACT_WINDOW)
                if mask_e[idx_obs].all() and mask_e[idx_act].all():
                    self.samples.append((
                        obs_seq_e[idx_obs].astype(np.float32),
                        act_seq_e[idx_act].astype(np.float32)
                    ))
        if not self.samples:
            raise RuntimeError(f"No valid training windows found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_seq, act_seq = self.samples[idx]
        return (
            torch.from_numpy(obs_seq),
            torch.from_numpy(act_seq)
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
    offset_loss_multiplier=1e3,
    obs_window_size=N_OBS_WINDOW,
    act_window_size=N_ACT_WINDOW
).to(DEVICE)

# Optimizer
optim = AdamW(agent.parameters(), lr=TRANSFORMER_LR)

# ─────────────── DataLoader & Training Loop ───────────────
if __name__ == '__main__':
    train_ds = UR3SequenceDataset(DATA_DIR)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        '''
        for obs_seq, act_seq in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            obs_seq = obs_seq.to(DEVICE)
            act_seq = act_seq.to(DEVICE)
            predicted, _, metrics = agent(obs_seq, None, act_seq)   # predicted: (B*N_OBS_WINDOW, ACT_DIM)
            # flatten your ground‐truth windows in the same order
            target = act_seq.view(-1, ACT_DIM)                       # shape (B*N_OBS_WINDOW, ACT_DIM)
            # compute offset‐loss yourself so it lines up
            
            offset_loss = F.l1_loss(predicted, target, reduction="mean")
            # pull out the code‐classification loss that the agent already computed
            class_loss = metrics["code_loss"]                        # or whatever your key is
            loss = class_loss + offset_loss * agent.offset_loss_multiplier
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
            optim.step()
            total_loss += loss.item()
        '''
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
        print(f"[Epoch {epoch:03d}] Avg Loss: {avg:.4f}  CodeAcc: {metrics.get('code_acc',0):.3f}")
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            path = os.path.join(SAVE_DIR, f"agent_epoch{epoch:03d}.pt")
            torch.save(agent.state_dict(), path)
            print(f"--> Saved agent checkpoint: {path}")

    torch.save(agent.state_dict(), os.path.join(SAVE_DIR, "agent_final.pt"))
    print("Training complete. Model saved.")
