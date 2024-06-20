import os

import cv2
import numpy as np

link_pairs = [[0, 1], [1, 2], [2, 3], 
              [3, 4], [1, 5], [5, 6], 
              [6, 7], [1,8], [8, 9], 
              [9, 10], [1, 11], [11, 12],
              [12, 13],[0, 14],[0, 15], 
              [14, 16], [15, 17]]

link_pairs = [[0, 1], [0, 2], [1, 3], 
              [2, 4], [0, 17], [5, 17], 
              [5, 7], [7, 9], [6, 17], 
              [6, 8], [8, 10], [11, 17],
              [11, 13],[13, 15],[12, 17], 
              [12, 14], [14, 16]]

link_color = (255, 0, 0)

point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(128, 255, 0), (128, 0, 255)]

def vis_pose(image, pose_keypoints, show=True):

    if pose_keypoints[5][0] > 0 and pose_keypoints[6][0] > 0:
        neck = ((pose_keypoints[5][0] + pose_keypoints[6][0]) // 2, (pose_keypoints[5][1] + pose_keypoints[6][1]) // 2)
    else:
        neck = (-1, -1)
    pose_keypoints.append(neck)

    for idx, pair in enumerate(link_pairs):
        if pose_keypoints[pair[0]][0] < 0 or pose_keypoints[pair[1]][0] < 0:
            continue
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], link_color, 2)
    
    for idx, point in enumerate(pose_keypoints):
        if point[0] == -1:
            continue
        cv2.circle(image, point, 5, point_color[idx], thickness=-1)
        cv2.putText(image, f'{idx}', org=(point[0] - 5, point[1] - 5), fontFace=0, fontScale=0.5, color=point_color[idx], thickness=2)

    if show:
        cv2.imshow("image", image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def partition(np_list, width, height, show=True):
    partitioned_image = cv2.hconcat(np_list[:width])
    for i in range(width, np_list.shape[0], width):
        partitioned_image = cv2.vconcat([partitioned_image, cv2.hconcat(np_list[i:i + width])])
    if show:
        cv2.imshow('image', partitioned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return partitioned_image

def vis_clusters(image_size, cluster_keypoints_list, partition_size=(5, 6), show=True):
    np_list = []
    background = np.zeros(image_size)
    for cluster_center in cluster_keypoints_list:
        scaled_cluster_keypoints = [(int(x[0] * image_size[0]), int(x[1] * image_size[1])) for x in cluster_center]
        image = vis_pose(background.copy(), scaled_cluster_keypoints, show=False)
        np_list.append(image)
    np_list = np.array(np_list)
    partitioned_image = partition(np_list, partition_size[0], partition_size[1], show)
    return partitioned_image

if __name__ == '__main__':

    cluster_center_path = 'affordance_data/centers_30.txt'
    image_size = (256, 256, 3)

    cluster_center_list = []
    with open(cluster_center_path, 'r') as f:
        cluster_center_list = list(f.readlines())

    cluster_keypoints_list = []
    for cluster_data in cluster_center_list:
        cluster_data = cluster_data.split(' ')[:-1]
        cluster_data = [float(x) for x in cluster_data]
        cluster_keypoints = []
        for i in range(0, len(cluster_data), 2):
            cluster_keypoints.append((cluster_data[i], cluster_data[i+1]))
        cluster_keypoints = cluster_keypoints[:-1]
        cluster_keypoints_list.append(cluster_keypoints)

    vis_clusters(image_size, cluster_keypoints_list, partition_size=(5, 6), show=True)