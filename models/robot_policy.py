#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
# ── make sure Python can import detr/models ────────────────────────────────
sys.path.append(os.path.abspath(r'C:\Users\jupar\imitationLearning'))  # adjust to parent of models/
from models import Policy

# ── Dataset ─────────────────────────────────────────────────────────────────
class RobotEpisodeDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir,f)
                      for f in os.listdir(data_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as hf:
            img = hf['images'][()]   # (T,2,C,H,W)
            st  = hf['states'][()]   # (T,7)
        return torch.from_numpy(img), torch.from_numpy(st)

# ── VAE loss ────────────────────────────────────────────────────────────────
mse = MSELoss()
def vae_loss(recon, target, mu, logvar, kl_w):
    rec = mse(recon, target)
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return rec + kl_w * kl.mean()

# ── Main ────────────────────────────────────────────────────────────────────
def main(args):
    # device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & loader
    ds = RobotEpisodeDataset(args.data_dir)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=(args.mode=='train'),
                    drop_last=True)

    # instantiate model
    model = Policy(
        state_dim    = args.state_dim,
        camera_names = args.cameras
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        for ep in range(args.epochs):
            model.train()
            total = 0.0
            for imgs, states in dl:
                imgs, states = imgs.to(dev), states.to(dev)
                opt.zero_grad()
                recon, mu, lv = model(imgs, states)
                loss = vae_loss(recon, states, mu, lv, args.kl_weight)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total/len(dl)
            print(f'Epoch {ep+1}/{args.epochs} – loss {avg:.4f}')
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f'ckpt{ep+1:03d}.pth'))

    else:  # inference
        ckpt = torch.load(args.ckpt_path, map_location=dev)
        model.load_state_dict(ckpt)
        model.eval()
        os.makedirs(args.out_dir, exist_ok=True)
        with torch.no_grad():
            for i,(imgs,_) in enumerate(dl):
                imgs = imgs.to(dev)
                pred,_,_ = model(imgs)
                torch.save(pred.cpu(),
                           os.path.join(args.out_dir, f'pred_{i:03d}.pt'))
        print(f'Inference complete. Outputs in {args.out_dir}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train','infer'],
                   help='train or infer')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--state_dim', type=int, default=7)
    p.add_argument('--cameras', nargs='+', default=['cam0','cam1'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--kl_weight', type=float, default=1.0)
    p.add_argument('--ckpt_dir', default='./checkpoints')
    p.add_argument('--ckpt_path', help='for infer mode')
    p.add_argument('--out_dir', default='./predictions')
    args = p.parse_args()
    main(args)
