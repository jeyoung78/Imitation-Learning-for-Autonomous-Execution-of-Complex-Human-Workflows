import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def list_images(directory):
    """
    List image files (.jpg and .png) in a directory, sorted.
    """
    files = []
    for ext in ('*.jpg', '*.png'):
        files.extend(glob(os.path.join(directory, ext)))
    return sorted(files)


class AlohaMultiViewDataset(Dataset):
    """
    Dataset for multi-view ALOHA CVAE:
    - coords_list: numpy array of shape (N, seq_len, action_dim)
    - image_pairs: list of (path1, path2) tuples, length N
    """
    def __init__(self, coords_list, image_pairs, seq_len=20, transform=None):
        assert len(coords_list) == len(image_pairs), "coords and images lists must match"
        self.coords_list = coords_list
        self.image_pairs = image_pairs
        self.seq_len = seq_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        coords = self.coords_list[idx]  # (seq_len, action_dim)
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        imgs = torch.stack([img1, img2], dim=0)  # (2, C, H, W)
        return imgs, torch.from_numpy(coords).float()


class RecognitionEncoder(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, num_layers=4,
                 nhead=8, dim_feedforward=2048, max_seq_len=90):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.joint_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len+1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, coords):
        # coords: (B, seq_len, action_dim)
        B, L, _ = coords.shape
        x = self.joint_proj(coords)          # (B, L, H)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([cls, x], dim=1)       # (B, L+1, H)
        x = x + self.pos_emb[:, :L+1]        # add pos embeddings
        x = x.permute(1, 0, 2)               # (L+1, B, H)
        h = self.transformer(x)              # (L+1, B, H)
        cls_h = h[0]                         # (B, H)
        mu = self.mu_head(cls_h)
        logvar = self.logvar_head(cls_h)
        return mu, logvar


class MultiViewDecoder(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, num_layers=4,
                 nhead=8, dim_feedforward=2048, img_feat_dim=512,
                 num_views=2, chunk_len=10):
        super().__init__()
        self.num_views = num_views
        self.chunk_len = chunk_len
        # shared ResNet-18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.joint_proj = nn.Linear(action_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, hidden_dim)
        # separate positional embeddings
        self.enc_pos_emb = nn.Parameter(torch.randn(1, num_views + 2, hidden_dim))
        self.dec_pos_emb = nn.Parameter(torch.randn(1, chunk_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.out_mlp = nn.Linear(hidden_dim, action_dim)

    def forward(self, imgs, joint, z):
        # imgs: (B, num_views, C, H, W)
        B = imgs.size(0)
        # extract and project view features
        view_feats = []
        for v in range(self.num_views):
            feat = self.backbone(imgs[:, v])       # (B, img_feat_dim)
            feat = self.img_proj(feat)            # (B, H)
            view_feats.append(feat.unsqueeze(1))  # (B,1,H)
        feats = torch.cat(view_feats, dim=1)      # (B, num_views, H)
        joints = self.joint_proj(joint).unsqueeze(1)  # (B,1,H)
        z_tok = self.z_proj(z).unsqueeze(1)           # (B,1,H)
        enc_in = torch.cat([feats, joints, z_tok], dim=1)  # (B, num_views+2, H)
        enc_in = enc_in + self.enc_pos_emb[:, :self.num_views+2]
        enc = self.encoder(enc_in.permute(1,0,2))   # (num_views+2, B, H)
        # prepare repeated queries for decoder
        queries = self.dec_pos_emb.repeat(B, 1, 1)  # (B, chunk_len, H)
        queries = queries.permute(1,0,2)           # (chunk_len, B, H)
        dec_out = self.decoder(queries, enc)       # (chunk_len, B, H)
        dec_out = dec_out.permute(1,0,2)           # (B, chunk_len, H)
        actions = self.out_mlp(dec_out)            # (B, chunk_len, action_dim)
        return actions


class ALOHAMultiViewCVAE(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, num_layers=4,
                 nhead=8, dim_feedforward=2048, max_seq_len=20,
                 num_views=2, chunk_len=10):
        super().__init__()
        self.encoder = RecognitionEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len
        )
        self.decoder = MultiViewDecoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            img_feat_dim=512,
            num_views=num_views,
            chunk_len=chunk_len
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, imgs, coords):
        mu, logvar = self.encoder(coords)
        z = self.reparameterize(mu, logvar)
        joint = coords[:, -1]  # last timestep
        actions = self.decoder(imgs, joint, z)
        return actions, mu, logvar


def loss_function(recon_actions, true_actions, mu, logvar, beta=1.0):
    # Both inputs shape (B, chunk_len, action_dim)
    recon_loss = F.mse_loss(recon_actions, true_actions, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def train(args):
    print("Starting training…")
    coords = np.load(args.coords_path, allow_pickle=True)
    imgs1 = list_images(args.img_dir1)
    imgs2 = list_images(args.img_dir2)
    image_pairs = list(zip(imgs1, imgs2))
    dataset = AlohaMultiViewDataset(coords, image_pairs, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Found {len(dataset)} samples → {len(dataloader)} batches per epoch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ALOHAMultiViewCVAE(
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        nhead=args.heads,
        dim_feedforward=args.ffn,
        max_seq_len=args.seq_len,
        num_views=2,
        chunk_len=args.chunk_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, coords in dataloader:
            imgs, coords = imgs.to(device), coords.to(device)
            true = coords[:, -args.chunk_len:]
            optimizer.zero_grad()
            pred, mu, logvar = model(imgs, coords)
            loss, rl, kld = loss_function(pred, true, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Model saved to {args.model_out}")


def inference(args):
    print("Running inference…")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ALOHAMultiViewCVAE(
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        nhead=args.heads,
        dim_feedforward=args.ffn,
        max_seq_len=args.seq_len,
        num_views=2,
        chunk_len=args.chunk_len
    ).to(device)
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    model.eval()

    # load and preprocess images
    img1 = Image.open(args.img1_path).convert('RGB')
    img2 = Image.open(args.img2_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
    img1 = transform(img1)
    img2 = transform(img2)
    imgs = torch.stack([img1, img2], dim=0).unsqueeze(0).to(device)

    coords = np.load(args.init_coords).astype(np.float32)
    coords = torch.from_numpy(coords).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, _, _ = model(imgs, coords)
    print("Predicted actions:", pred.squeeze(0).cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ALOHA Multi-View CVAE Training & Inference'
    )
    sub = parser.add_subparsers(dest='mode')

    # training args
    t = sub.add_parser('train')
    t.add_argument('--coords_path', type=str, required=True)
    t.add_argument('--img_dir1',    type=str, required=True)
    t.add_argument('--img_dir2',    type=str, required=True)
    t.add_argument('--model_out',   type=str, default='outputs/aloha_model.pt')
    t.add_argument('--seq_len',     type=int, default=20)
    t.add_argument('--chunk_len',   type=int, default=10)
    t.add_argument('--action_dim',  type=int, default=14)
    t.add_argument('--hidden_dim',  type=int, default=512)
    t.add_argument('--layers',      type=int, default=4)
    t.add_argument('--heads',       type=int, default=8)
    t.add_argument('--ffn',         type=int, default=2048)
    t.add_argument('--batch_size',  type=int, default=4)
    t.add_argument('--epochs',      type=int, default=30)
    t.add_argument('--lr',          type=float, default=1e-4)
    t.add_argument('--beta',        type=float, default=1.0)

    # inference args
    i = sub.add_parser('infer')
    i.add_argument('--img1_path',   type=str, required=True)
    i.add_argument('--img2_path',   type=str, required=True)
    i.add_argument('--init_coords', type=str, required=True)
    i.add_argument('--model_out',   type=str, required=True)
    i.add_argument('--seq_len',     type=int, default=20)
    i.add_argument('--chunk_len',   type=int, default=10)
    i.add_argument('--action_dim',  type=int, default=14)
    i.add_argument('--hidden_dim',  type=int, default=512)
    i.add_argument('--layers',      type=int, default=4)
    i.add_argument('--heads',       type=int, default=8)
    i.add_argument('--ffn',         type=int, default=2048)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        inference(args)
    else:
        parser.print_help()