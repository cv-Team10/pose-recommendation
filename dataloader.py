import os

import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MaskDataset(Dataset):
    def __init__(self, data_path, data_list, cfg):

        self.cfg = cfg
        self.image_resize = self.cfg.image_resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_resize),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_path = data_path
        self.image_dir_path = os.path.join(self.data_path, 'image')
        self.annotation_list = self.preprocess(data_list)

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image_name, norm_pose_keypoints, class_label, mask_box = self.annotation_list[idx]

        cv_image = cv2.imread(os.path.join(self.image_dir_path, image_name))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image[mask_box[0]:mask_box[2], mask_box[1]:mask_box[3], :] = 0
        
        transformed_image = self.transform(cv_image)
        norm_pose_keypoints = torch.tensor(norm_pose_keypoints)
        class_label = torch.tensor(class_label)

        return transformed_image, norm_pose_keypoints, class_label, image_name, torch.tensor(mask_box)
    
    def preprocess(self, data_list):
        annotation_list = []
        for data in data_list:
            splited_data = data.split('\t')
            image_name = splited_data[0]
            norm_pose_keypoints = eval(splited_data[1])
            class_label = eval(splited_data[2])
            mask_box = eval(splited_data[3])
            annotation_list.append((image_name, norm_pose_keypoints, class_label, mask_box))
        return annotation_list
    
def convert_norm_to_keypoints(norm_keypoints, label, mask_box, threshold=0.5):
        mask_size = (mask_box[2] - mask_box[0], mask_box[3] - mask_box[1])
        keypoints = [(round(x[0] * mask_size[0] + mask_box[0]), round(x[1] * mask_size[1] + mask_box[1])) if label[idx] > threshold else (-1, -1) for idx, x in enumerate(norm_keypoints)]
        return keypoints

if __name__ == '__main__':

    data_list = []
    with open('./mask_data.csv', 'r') as f:
        data_list = list(f.readlines())
    dataset = MaskDataset('./', data_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    annotation_list = dataset.annotation_list
    for annotation in annotation_list:
        image_name = annotation[0]
        norm_keypoints = annotation[1]
        label = annotation[2]
        mask_box = annotation[3]
        keypoints = convert_norm_to_keypoints(norm_keypoints, label, mask_box)
        print(image_name)
        print(keypoints)
        input()