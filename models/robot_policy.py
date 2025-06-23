#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss

# ── allow import of your detr/models folder ─────────────────────────────────
sys.path.append(os.path.abspath(r'/home/jhkim/Downloads/imit/Imitation-Learning-for-Autonomous-Execution-of-Complex-Human-Workflows'))  # ← parent of models/
from models.detr_vae import build as Policy

import inspect
print(inspect.signature(Policy))

# ── CSV‐backed Dataset ──────────────────────────────────────────────────────
class RobotCSV_Dataset(Dataset):
    def __init__(self, csv_dir):
        self.files = []
        for root, _, files in os.walk(csv_dir):
            for f in files:
                if f.endswith('.csv'):
                    self.files.append(os.path.join(root, f))
        if not self.files:
            raise RuntimeError(f"No CSV files found under {csv_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load and parse
        df = pd.read_csv(self.files[idx],
                         parse_dates=['robot_ts','cam0_ts','cam1_ts'])

        imgs, states = [], []
        for _, row in df.iterrows():
            # 1) load images, normalize to [0,1], shape C×H×W
            img0 = np.array(Image.open(row['cam0_path']),dtype=np.float32).transpose(2,0,1)/255.0
            img1 = np.array(Image.open(row['cam1_path']),dtype=np.float32).transpose(2,0,1)/255.0
            imgs.append([img0, img1])

            # 2) extract your 6–7 state dims
            state = np.array([
                row['x'], row['y'], row['z'],
                row['w'], row['p'], row['r'], row['ee'],
                # if you have a 7th joint do: row['your_7th_col']
    # ← example filler if you only have 6 joints
            ], dtype=np.float32)
            states.append(state)

        imgs   = torch.from_numpy(np.stack(imgs,   axis=0))
        states = torch.from_numpy(np.stack(states, axis=0))
        return imgs, states

# ── VAE loss ────────────────────────────────────────────────────────────────
mse = MSELoss()
def vae_loss(recon, target, mu, logvar, kl_w):
    rec = mse(recon, target)
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return rec + kl_w * kl.mean()

def main(args):
    import os
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import Adam

    # assume RobotCSV_Dataset, vae_loss, and Policy (build) are already imported above

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) DataLoader
    dataset = RobotCSV_Dataset(args.data_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.mode == 'train'),
        drop_last=True
    )

    # 2) Model + optimizer
    model     = Policy(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        os.makedirs(args.ckpt_dir, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for imgs, states in loader:
                # to device
                imgs, states = imgs.to(device), states.to(device)
                # print(states)
                # print(imgs)
                print(epoch)

                # flatten time: (B,T,2,C,H,W)→(B*T,2,C,H,W) & (B,T,D)→(B*T,D)
                B, T, V, C, H, W = imgs.shape
                imgs   = imgs.view(B*T, V, C, H, W)
                states = states.view(B*T, args.state_dim)

                optimizer.zero_grad()

                # ← correct keyword args here:
                recon, mu, logvar = model(
                    qpos      = states,
                    image     = imgs,
                    env_state = states
                )

                loss = vae_loss(recon, states, mu, logvar, args.kl_weight)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg = total_loss / len(loader)
            print(f"Epoch {epoch}/{args.epochs} — loss {avg:.4f}")
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'ckpt{epoch:03d}.pth'))

    else:  # infer
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        os.makedirs(args.out_dir, exist_ok=True)

        with torch.no_grad():
            for i, (imgs, states) in enumerate(loader):
                imgs, states = imgs.to(device), states.to(device)

                # flatten time
                B, T, V, C, H, W = imgs.shape
                imgs   = imgs.view(B*T, V, C, H, W)
                states = states.view(B*T, args.state_dim)

                # ← and here too:
                pred, mu, logvar = model(
                    qpos      = states,
                    image     = imgs,
                    env_state = states
                )

                torch.save(pred.cpu(),
                           os.path.join(args.out_dir, f'pred_{i:03d}.pt'))

        print(f'Inference complete → {args.out_dir}')

'''
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # swap in our CSV dataset
    ds = RobotCSV_Dataset(args.data_dir)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=(args.mode=='train'),
                    drop_last=True)

    model = Policy(
        state_dim    = args.state_dim,
        camera_names = args.camera_names
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        os.makedirs(args.ckpt_dir, exist_ok=True)
        for ep in range(args.epochs):
            model.train()
            total_loss = 0.0
            for imgs, states in dl:
                imgs, states = imgs.to(device), states.to(device)
                opt.zero_grad()
                recon, mu, lv = model(imgs, states)
                loss = vae_loss(recon, states, mu, lv, args.kl_weight)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg = total_loss / len(dl)
            print(f'Epoch {ep+1}/{args.epochs} — loss {avg:.4f}')
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'ckpt{ep+1:03d}.pth'))

    else:  # infer
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        os.makedirs(args.out_dir, exist_ok=True)
        with torch.no_grad():
            for i,(imgs,_) in enumerate(dl):
                imgs = imgs.to(device)
                pred,_,_ = model(imgs)
                torch.save(pred.cpu(),
                           os.path.join(args.out_dir, f'pred_{i:03d}.pt'))
        print(f'Inference complete → {args.out_dir}')
'''
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train','infer'],
               help='train or infer')
    p.add_argument('--data_dir',   required=True,
                help='folder containing your CSV episodes')
    p.add_argument('--state_dim',  type=int,   default=7,
                help='number of state dims per timestep')
    p.add_argument('--camera_names', nargs='+', default=['cam0','cam1'],
                help='names of camera streams, e.g. cam0 cam1')
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--lr',         type=float, default=1e-4,
                help='learning rate for VAE/policy head')
    p.add_argument('--kl_weight',  type=float, default=1.0)
    p.add_argument('--ckpt_dir',   type=str,   default='./checkpoints')
    p.add_argument('--ckpt_path',  type=str,
                help='path to checkpoint (required for infer mode)')
    p.add_argument('--out_dir',    type=str,   default='./predictions',
                help='where to save inference outputs')

    # ── DETR-VAE / transformer args ──────────────────────────────────────────────
    p.add_argument('--hidden_dim',      type=int,   default=128,
                help='dimension of transformer embeddings')
    p.add_argument('--dim_feedforward', type=int,   default=512,
                help='feed-forward network hidden dimension')
    p.add_argument('--nheads',          type=int,   default=8,
                help='number of attention heads')
    p.add_argument('--dropout',         type=float, default=0.1,
                help='dropout rate in transformer')
    p.add_argument('--position_embedding',
                choices=['sine','v2'], default='sine',
                help='type of positional encoding')
    p.add_argument('--enc_layers',      type=int,   default=6,
                help='number of encoder layers')
    p.add_argument('--dec_layers',      type=int,   default=6,
                help='number of decoder layers')
    p.add_argument('--pre_norm',        action='store_true',
                help='use pre-norm (normalize_before) in transformer')
    p.add_argument('--num_queries',     type=int,   default=100,
                help='number of object queries for DETR')

    # ── backbone & mask-head args ─────────────────────────────────────────────────
    p.add_argument('--backbone',    type=str,   default='resnet50',
                help='backbone CNN (e.g. resnet50)')
    p.add_argument('--dilation',    action='store_true',
                help='use dilated backbone')
    p.add_argument('--lr_backbone', type=float, default=0.0,
                help='learning rate for backbone; ≤0 to freeze')
    p.add_argument('--masks',       action='store_true',
                help='return intermediate layers for mask head')

    args = p.parse_args()
    main(args)
