import argparse
import os 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from torchvision import models, transforms

class RecognitionEncoder(nn.Module):
    def __init__(self, action_dim=14, joint_dim=14, hidden_dim=512,
                 num_layers=4, nhead=8, dim_feedforward=3200, max_seq_len=90):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len+2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, actions, joints):
        B, K, _ = actions.shape
        cls = self.cls_token.expand(B, -1, -1)
        j_emb = self.joint_proj(joints).unsqueeze(1)
        a_emb = self.action_proj(actions.view(B*K, -1)).view(B, K, -1)
        tokens = torch.cat([cls, j_emb, a_emb], dim=1)
        tokens = tokens + self.pos_emb[:, :K+2]
        tokens = tokens.transpose(0,1)
        out = self.transformer(tokens)                             
        cls_out = out[0]
        mu = self.mu_head(cls_out)
        logvar = self.logvar_head(cls_out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
class CVAE(nn.Module):
    def __init__(self, **kwargs):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder = RecognitionEncoder(**kwargs)
    
    def encode(self, actions, joints):
        return self.encoder(actions, joints)
    
    def decode(self, images, joints, z, actions_gt=None):
        return self.decoder(images, joints, z, targets=actions_gt)