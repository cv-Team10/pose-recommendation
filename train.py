from __future__ import print_function
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from dataloader import MaskDataset
from loss import CustomLoss
from pose_resnet import *
from resnet import MultiHeadResNet34
from transformer import KeypointAttention, Backbone

def train_epoch(train_loader, criterion, optimizer):

    total_loss = []
    for image, norm_pose_keypoints, class_label, _, _ in train_loader:

        # Forward pass
        outputs = model(image.to(device))

        targets = torch.cat((norm_pose_keypoints, class_label.unsqueeze(-1)), dim=-1)
        # Compute the loss
        loss = criterion(outputs.to(device), targets.to(device))
        print(loss)
        total_loss.append(loss)
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()

    return np.mean(total_loss)

if __name__ == '__main__':
    cfg = Config()

    device = cfg.device

    if cfg.model_type == 'pose_resnet':
        model = PoseResNet(Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(torch.load("./pretrained/pose_resnet_50_256x192.pth.tar", map_location=device),strict=False)
        custom_layer = model.custom_layer  
        for param in custom_layer.parameters():
            param.data.fill_(0.05)
    if cfg.model_type == 'resnet':
        model = MultiHeadResNet34()
    if cfg.model_type == 'transformer':
        backbone = Backbone()
        model = KeypointAttention(backbone, feature_dim = 256, num_queries = 17, num_attention_layers = 6, num_heads = 4)

    model.train()
    model.to(device)

    data_path = './data'
    data_list = []                  
    with open(os.path.join(data_path, 'mask_data.csv'), 'r') as f:
        data_list = list(f.readlines())

    train_dataset = MaskDataset('./data', data_list, cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        loss = train_epoch(train_dataloader, criterion, optimizer)

        # Optionally, print the loss after each epoch
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'saved_model_{epoch}_{loss}.pth'))