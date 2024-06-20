import os

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataloader import MaskDataset


if __name__ == '__main__':

    cfg = Config()

    threshold = 0.4

    data_path = './data'
    image_path = os.path.join(data_path, 'image')
    data_list = []                  
    with open(os.path.join(data_path, 'mask_data.csv'), 'r') as f:
        data_list = list(f.readlines())

    train_dataset = MaskDataset('./data', data_list, cfg)
    annotation_list = train_dataset.preprocess(data_list)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    total_len = train_dataset.__len__()
    correct_len = 0

    for image_name, norm_pose_keypoints, class_label, mask_box in tqdm(annotation_list):
        mask_area = (mask_box[3] - mask_box[1]) * (mask_box[2] - mask_box[0])
        image = Image.open(os.path.join(image_path, image_name))
        image_size = image.size
        image_area = image_size[0] * image_size[1]
        ratio = mask_area / image_area
        if ratio < threshold:
            correct_len += 1
            with open(f'mask_{threshold}.txt', 'a') as f:
                f.writelines(f'{image_name}\n')
    print(correct_len / total_len)
    print(correct_len / total_len * total_len)