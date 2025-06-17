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

# Loads images and corresponding coordinate sequences
# coords_list: list of numpy arrays of shape (seq_len, action_dim)
# image_paths: list of image file paths
class Dataset(Dataset):
    def __init__(self, coords_list, image_paths, seq_len=10, transform=None):
        assert len(coords_list) == len(image_paths), "Mismatched data lengths"
        self.coords_list = coords_list
        self.image_paths = image_paths
        self.seq_len = seq_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((480, 640)),  
            transforms.ToTensor(),          
        ])

        def __len__(self):
            return len(self.coords_list)

        def __getitem__(self, idx):
            coords = self.coords_list[idx]  # (seq_len, action_dim)
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(coords).float()
        
# Encoder part of the CVAE
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