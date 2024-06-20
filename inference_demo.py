import os

import cv2
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
from torchvision import transforms
from vae import VAE
from transformer import KeypointAttention, Backbone

def convert_norm_to_keypoints(norm_keypoints, label, mask_box, threshold=0.5):
    mask_size = (mask_box[2] - mask_box[0], mask_box[3] - mask_box[1])
    keypoints = [(round(x[0] * mask_size[0] + mask_box[0]), round(x[1] * mask_size[1] + mask_box[1])) if label[idx] > threshold else (-1, -1) for idx, x in enumerate(norm_keypoints)]
    return keypoints

image = None
copied_image = None
mask_box = [0, 0, 0, 0]
click = False

def draw_rectangle(event, x, y, flags, param):
    global mask_box, click, copied_image

    if event == cv2.EVENT_LBUTTONDOWN:
        click = True
        copied_image = image.copy()
        mask_box[0] = y
        mask_box[1] = x
		
    elif event == cv2.EVENT_MOUSEMOVE:
        if click == True:
            cv2.rectangle(copied_image, (mask_box[1], mask_box[0]), (x, y), (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        click = False
        mask_box[2] = y
        mask_box[3] = x
        cv2.rectangle(copied_image,(mask_box[1], mask_box[0]),(x, y), (0, 0, 0), -1)
    cv2.imshow('image', copied_image)

def get_mask_box(image_path):
    global image
    global copied_image
    image = cv2.imread(image_path)
    copied_image = image.copy()
    
    cv2.imshow('image', copied_image)
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        k = cv2.waitKey(0)
        if k == 13: # press ENTER
            cv2.destroyAllWindows()
            break
    return mask_box

if __name__ == '__main__':
        
    cfg = Config()

    device = 'cpu'

    model_path = 'checkpoints/experiment4/saved_model_v1_dvcoef0.02_35_0.124,-1.103.pth'

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

    image_name = 'test_image12.png'
    image_path = os.path.join('test', image_name)

    mask_box = get_mask_box(image_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.image_resize),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        cv_image = cv2.imread(image_path)
        cv_image[mask_box[0]:mask_box[2], mask_box[1]:mask_box[3], :] = 0
        cv2.imshow('image', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        transformed_image = transform(cv_image)

        if cfg.model_type == 'vae':
            z = torch.randn((1, 30))
            outputs = model.decode(z, transformed_image.unsqueeze(0).to(device))
        else:
            outputs = model(transformed_image.unsqueeze(0).to(device))
        print(outputs)
        keypoints = convert_norm_to_keypoints(outputs[0, :, :2].tolist(), outputs[0, :, 2].tolist(), mask_box, threshold=0.8)
        image_name = image_name[0]
        mask_box = mask_box
        print(keypoints)
        vis_pose(image_path, keypoints, mask_box)
