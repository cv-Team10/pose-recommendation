import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import convert_series_to_keypoints, preprocess_keypoints_for_clustering, write_cluster_centers, write_pose_data, write_cluster_cnt, get_cluster_center_by_choice
from vis_clusters import vis_clusters

k_cluster = 30
partition_size = (5, 6)
random_state = 0
target_data_path = 'data'
if not os.path.exists(target_data_path):
    os.mkdir(target_data_path)

data_path = './data/mask_data.csv'
image_name_list, norm_keypoints_list = preprocess_keypoints_for_clustering(data_path)
kmeans = KMeans(n_clusters=k_cluster, random_state=random_state, n_init="auto").fit(norm_keypoints_list)

cluster_center = kmeans.cluster_centers_
labels = kmeans.labels_

cluster_center_keypoints = []
for center in cluster_center:
    keypoints = []
    for i in range(0, len(center), 2):
        keypoints.append((center[i+1], center[i]))
    cluster_center_keypoints.append(keypoints)
vis_clusters((256, 256, 3), np.array(cluster_center_keypoints), partition_size, show=True)

"""
write_cluster_centers(os.path.join(target_data_path, f'centers_{k_cluster}.txt'), cluster_center)
# write_pose_data(os.path.join(target_data_path, f'trainlist{tag}.txt'), image_name_list, keypoints_list, labels)
write_cluster_cnt(os.path.join(target_data_path, f'cluster_counts{tag}.txt'), labels)

keypoints_center_list = []
for center in cluster_center:
    keypoints_center_list.append(convert_series_to_keypoints(center))
cluster_center_image = vis_clusters(image_size=(256, 256, 3), cluster_keypoints_list=keypoints_center_list, partition_size=partition_size)
cv2.imwrite(os.path.join(target_data_path, f'cluster_centers{tag}.jpg'), cluster_center_image)

data_path = './affordance_data/testlist.txt'
image_name_list, keypoints_list, norm_keypoints_list = preprocess_keypoints_for_clustering(data_path)

labels = kmeans.predict(np.array(norm_keypoints_list))
write_pose_data(os.path.join(target_data_path, f'testlist{tag}.txt'), image_name_list, keypoints_list, labels)
"""