import os
import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data

import sys, pathlib
# make the repo root importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from vqvae.vqvae import VqVae
from vq_behavior_transformer.gpt import GPT, GPTConfig
from vq_behavior_transformer.bet_sd import BehaviorTransformer
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random
import tqdm
from pathlib import Path

EPOCHS = 30
BATCH_SIZE = 1
N_OBS_WINDOW       = 1          
N_ACT_WINDOW       = 1          
OBS_DIM            = 31 
DATA_DIR = "./data/button"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VqVae(
    input_dim_h=1,      # ACTION_WINDOW_SIZE
    input_dim_w=10,     # ACT_DIM
    n_latent_dims=512,  # N_LATENT_DIMS
    vqvae_n_embed=32,   # VQVAE_N_EMBED
    vqvae_groups=2,     # VQVAE_GROUPS
    eval=True,
    device=DEVICE
)
vqvae.load_state_dict(torch.load("./checkpoints_pybullet/buttons/more_more_data/vqvae/vqvae_epoch200.pt", map_location=DEVICE))

small_cfg = GPTConfig(
    input_dim=31,
    output_dim=32*2,
    block_size=1,
    n_layer=6,        # smaller
    n_head=8,
    n_embd=128,       # smaller
    dropout=0.1
)
gpt_small = GPT(small_cfg).to(DEVICE)
agent_small = BehaviorTransformer(
    obs_dim=31, act_dim=10, goal_dim=0,
    gpt_model=gpt_small, vqvae_model=vqvae,
    offset_loss_multiplier=1e1,
    obs_window_size=1, act_window_size=1
).to(DEVICE)
checkpoint = torch.load("./checkpoints_pybullet/buttons/more_data/transformer/agent_epoch200.pt", map_location=DEVICE)
agent_small.load_state_dict(checkpoint)
agent_small.eval()

large_cfg = GPTConfig(
    input_dim=31,
    output_dim=32*2,
    block_size=1,
    n_layer=12,        # smaller
    n_head=8,
    n_embd=256,       # smaller
    dropout=0.1
)
# — large model (full), as before
gpt_large = GPT(large_cfg).to(DEVICE)
agent_large = BehaviorTransformer(
    obs_dim=31, act_dim=10, goal_dim=0,
    gpt_model=gpt_large, vqvae_model=vqvae,
    offset_loss_multiplier=1e1,
    obs_window_size=1, act_window_size=1
).to(DEVICE)
checkpoint = torch.load("./checkpoints_pybullet/buttons/more_more_data/transformer/agent_epoch1600.pt", map_location=DEVICE)
agent_large.load_state_dict(checkpoint)
agent_large.eval()

'''
obs_window = torch.tensor(
    [[[-5.1341e-03, -2.2752e-03, -4.4743e-03, -6.6800e-03, -3.1875e-03,
        9.3089e-03, -2.5150e-03, -2.4857e-03,  3.8110e-04, -2.5272e-04,
       -8.3230e-01, -3.2338e-01, -7.4506e-01, -1.0144e+00, -5.0301e-01,
        1.4117e+00, -3.7833e-01, -3.7402e-01,  6.7229e-02, -3.0405e-02,
        2.2465e-02,  2.3868e-02,  1.4898e+00,  9.3322e-01, -2.4749e-01,
        2.9083e-02,  9.4557e-01, -3.9408e-01,  3.6280e-02,  4.0000e-03,
        0.0000e+00]]],
    device='cuda:0'
)
'''


# probs_small, cbet_offsets_small = agent_small(obs_window, None, None, False, None, None)
# print(probs_small)

# probs_large, cbet_offsets_large = agent_large(obs_window, None, None, False, None, None)
# print(probs_large)

def speculative_decode_probs(p_small: torch.Tensor,
                            p_large: torch.Tensor,
                            k: int = 1) -> torch.LongTensor:
    B, G, C = p_small.shape
    selected = torch.zeros(B, G, dtype=torch.long, device=p_small.device)

    # 2) for each batch and group
    for b in range(B):
        for g in range(G):
            # draw up to k proposals from the small model
            proposals = torch.multinomial(p_small[b, g], num_samples=k, replacement=True)  # [k]
            # test each proposal
            for idx in proposals:
                ps = p_small[b, g, idx]
                pr = p_large[b, g, idx]
                alpha = min(1.0, (pr / ps).item())  # acceptance ratio
                if torch.rand((), device=p_small.device).item() < alpha:
                    selected[b, g] = idx
                    break
            else:
                # none accepted → sample from the refine model
                selected[b, g] = torch.multinomial(p_large[b, g], num_samples=1).squeeze(0)

    return selected

# spec_sampled = speculative_decode_probs(probs_small, probs_large)

# probs_large, loss, metrics = agent_large(obs_window, None, None, True, spec_sampled, cbet_offsets_large)
# print(loss)

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

def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

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

    for epoch in range(1, EPOCHS+1):
            total_loss = 0.0
            count = 0
            for obs_seq, act_seq in train_loader:
                obs_seq = obs_seq.to(DEVICE)
                act_seq = act_seq.to(DEVICE)
                
                # agent returns (predictions, total_loss, metrics_dict)
                agent_small.return_probs = True
                agent_large.return_probs = True
                probs_small, cbet_offsets_small, cbet_logits_small = agent_small(obs_seq, None, None, False, None, None, None)
                probs_large, cbet_offsets_large, cbet_logits_large = agent_large(obs_seq, None, None, False, None, None, None)
                spec_sampled = speculative_decode_probs(probs_small, probs_large)

                agent_small.return_probs = False
                agent_large.return_probs = False
                action, _, _ = agent_small(obs_seq, None, None, True, spec_sampled, cbet_offsets_large, None)

                mse = F.mse_loss(action, act_seq, reduction='mean')
                total_loss += mse.item()
                count += 1
                print(total_loss/count)
                

            avg = total_loss / len(train_loader)
            print(avg)