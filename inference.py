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
from vis_pose import vis_pose
from transformer import KeypointAttention, Backbone
from vae import VAE

def convert_norm_to_keypoints(norm_keypoints, label, mask_box, threshold=0.5):
    mask_size = (mask_box[2] - mask_box[0], mask_box[3] - mask_box[1])
    keypoints = [(round(x[0] * mask_size[0] + mask_box[0]), round(x[1] * mask_size[1] + mask_box[1])) if label[idx] > threshold else (-1, -1) for idx, x in enumerate(norm_keypoints)]
    return keypoints

if __name__ == '__main__':
        
    cfg = Config()

    device = 'cpu'

    model_path = 'checkpoints/experiment3/saved_model_29_0.23590107262134552.pth'

    if cfg.model_type == 'pose_resnet':
        model = PoseResNet(Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
        custom_layer = model.custom_layer  
        for param in custom_layer.parameters():
            param.data.fill_(0.05)
    if cfg.model_type == 'resnet':
        model = MultiHeadResNet34()
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    if cfg.model_type == 'transformer':
        backbone = Backbone()
        model = KeypointAttention(backbone, feature_dim = 256, num_queries = 17, num_attention_layers = 6, num_heads = 4)
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    if cfg.model_type == 'vae':
        model = VAE('VGG')
        model.load_state_dict(torch.load('checkpoints/experiment4/saved_model_vae_11_0.1991470018593372.pth', map_location=device))

    model.eval
    model.to(device)

    data_path = './data'
    image_path = os.path.join(data_path, 'image')
    data_list = []                  
    with open(os.path.join(data_path, 'mask_data.csv'), 'r') as f:
        data_list = list(f.readlines())

    train_dataset = MaskDataset('./data', data_list, cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    with torch.no_grad():
        for image, norm_pose_keypoints, class_label, image_name, mask_box in train_dataloader:

            if cfg.model_type == 'vae':
                z = torch.randn((1, 30))
                outputs = model.decode(z, image.to(device))
            else:
                outputs = model(image.to(device))
            print(outputs)
            keypoints = convert_norm_to_keypoints(outputs[0, :, :2].tolist(), outputs[0, :, 2].tolist(), mask_box[0].tolist(), threshold=0.8)
            gt_keypoints = convert_norm_to_keypoints(norm_pose_keypoints[0, :, :2].tolist(), class_label[0, :].tolist(), mask_box[0].tolist(), threshold=0.8)
            image_name = image_name[0]
            mask_box = mask_box[0]
            print(keypoints)
            # vis_pose(os.path.join(image_path, image_name), gt_keypoints, mask_box)
            vis_pose(os.path.join(image_path, image_name), gt_keypoints, [0, 0, 0, 0])
