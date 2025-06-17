import os  # for file path operations
import argparse  # for CLI argument parsing
from glob import glob  # to list files in a directory
from PIL import Image  # to load images
import numpy as np  # for numerical operations
import torch  # PyTorch core
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # functional API for losses, activations
from torch.utils.data import Dataset, DataLoader  # data handling
from torchvision import transforms, models  # data transforms and pretrained models


class AlohaMultiViewDataset(Dataset):
    """
    Dataset for ALOHA CVAE with multi-view: loads two images and corresponding coordinate sequences.
    coords_list: list of numpy arrays of shape (seq_len, action_dim)
    image_pairs: list of tuples (path_view1, path_view2)
    """
    def __init__(self, coords_list, image_pairs, seq_len=10, transform=None):
        assert len(coords_list) == len(image_pairs), "Mismatched data lengths"
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
        coords = self.coords_list[idx]
        img1 = Image.open(self.image_pairs[idx][0]).convert('RGB')
        img2 = Image.open(self.image_pairs[idx][1]).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        imgs = torch.stack([img1, img2], dim=0)  # shape (2, C, H, W)
        return imgs, torch.from_numpy(coords).float()


class RecognitionEncoder(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, num_layers=4, nhead=8, dim_feedforward=2048, max_seq_len=90):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.joint_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len+1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, coords):
        B, L, _ = coords.shape
        x = self.joint_proj(coords)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :L+1]
        x = x.permute(1, 0, 2)
        h = self.transformer(x)
        cls_h = h[0]
        mu = self.mu_head(cls_h)
        logvar = self.logvar_head(cls_h)
        return mu, logvar


class MultiViewDecoder(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, num_layers=4, nhead=8, dim_feedforward=2048,
                 img_feat_dim=512, num_views=2, chunk_len=10):
        super().__init__()
        self.num_views = num_views
        # shared ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.joint_proj = nn.Linear(action_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_views + 2, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.out_mlp = nn.Linear(hidden_dim, action_dim)
        self.chunk_len = chunk_len

    def forward(self, imgs, joint, z):
        B = imgs.size(0)
        view_feats = []
        for v in range(self.num_views):
            feat = self.backbone(imgs[:, v])
            feat = self.img_proj(feat)
            view_feats.append(feat.unsqueeze(1))
        feats = torch.cat(view_feats, dim=1)  # (B, num_views, H)
        joints = self.joint_proj(joint).unsqueeze(1)  # (B,1,H)
        z_tok = self.z_proj(z).unsqueeze(1)  # (B,1,H)
        enc_in = torch.cat([feats, joints, z_tok], dim=1)  # (B, num_views+2, H)
        enc_in = enc_in + self.pos_emb[:, :self.num_views+2]
        enc = self.encoder(enc_in.permute(1,0,2))  # (num_views+2, B, H)
        queries = self.pos_emb[:, :self.chunk_len]  # (1,chunk_len,H)
        dec_out = self.decoder(queries.permute(1,0,2), enc)  # (chunk_len, B, H)
        dec_out = dec_out.permute(1,0,2)
        actions = self.out_mlp(dec_out)
        return actions


class ALOHAMultiViewCVAE(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=512, **kwargs):
        super().__init__()
        self.encoder = RecognitionEncoder(action_dim=action_dim, hidden_dim=hidden_dim, **kwargs)
        self.decoder = MultiViewDecoder(action_dim=action_dim, hidden_dim=hidden_dim, **kwargs)
        self.hidden_dim = hidden_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, imgs, coords):
        mu, logvar = self.encoder(coords)
        z = self.reparameterize(mu, logvar)
        joint = coords[:, -1]
        actions = self.decoder(imgs, joint, z)
        return actions, mu, logvar


def loss_function(recon_actions, true_actions, mu, logvar, beta=1.0):
    recon_loss = F.l1_loss(recon_actions, true_actions, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def train(args):
    coords_data = np.load(args.coords_path, allow_pickle=True)
    # expect two folders: img_dir1 and img_dir2 containing matching image files
    imgs1 = sorted(glob(os.path.join(args.img_dir1, '*.jpg')))
    imgs2 = sorted(glob(os.path.join(args.img_dir2, '*.jpg')))
    image_pairs = list(zip(imgs1, imgs2))
    dataset = AlohaMultiViewDataset(coords_data, image_pairs, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, coords in dataloader:
            imgs, coords = imgs.to(device), coords.to(device)
            true = coords[:, -args.chunk_len:]
            optimizer.zero_grad()
            pred, mu, logvar = model(imgs, coords)
            loss, rl, kld = loss_function(pred, true, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), args.model_out)


def inference(args):
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

    img1 = Image.open(args.img1_path).convert('RGB')
    img2 = Image.open(args.img2_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
    img1 = transform(img1)
    img2 = transform(img2)
    imgs = torch.stack([img1, img2], dim=0).unsqueeze(0).to(device)  # (1,2,C,H,W)

    coords = np.load(args.init_coords).astype(np.float32)
    coords = torch.from_numpy(coords).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, _, _ = model(imgs, coords)
    print('Predicted actions:', pred.squeeze(0).cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALOHA Multi-View CVAE Training & Inference')
    sub = parser.add_subparsers(dest='mode')

    t = sub.add_parser('train')
    t.add_argument('--coords_path', type=str, required=True)
    t.add_argument('--img_dir1', type=str, required=True)
    t.add_argument('--img_dir2', type=str, required=True)
    t.add_argument('--model_out', type=str, default='aloha_cvae.pt')
    t.add_argument('--seq_len', type=int, default=20)
    t.add_argument('--chunk_len', type=int, default=10)
    t.add_argument('--action_dim', type=int, default=14)
    t.add_argument('--hidden_dim', type=int, default=512)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--heads', type=int, default=8)
