import numpy as np

def convert_series_to_keypoints(series):
    keypoints = []
    for i in range(0, len(series), 2):
        keypoints.append((series[i], series[i+1]))
    return np.array(keypoints)

def norm_pose(pose_keypoints_series):
    pose_keypoints = np.array(pose_keypoints_series)
    width = np.max(pose_keypoints[0::2]) - np.min(pose_keypoints[0::2])
    height = np.max(pose_keypoints[1::2]) - np.min(pose_keypoints[1::2])
    min_x = np.min(pose_keypoints[0::2])
    min_y = np.min(pose_keypoints[1::2])

    norm_pose_keypoints = []
    for idx, point in enumerate(pose_keypoints):
        if idx % 2 == 0:
            point = (point - min_x) / width
        else:
            point = (point - min_y) / height
        norm_pose_keypoints.append(point)
    return norm_pose_keypoints

def preprocess_keypoints_for_clustering(data_path):
    with open(data_path, 'r') as f:
        data_list = list(f.readlines())
    data_list = [x.split('\t') for x in data_list]

    image_name_list = []
    norm_keypoints_list = []
    for data in data_list:
        image_name = data[0]
        keypoint = eval(data[1])
        label = eval(data[2])
        preprocessed_keypoints = []
        for idx, point in enumerate(keypoint):
            if label[idx] == 0:
                preprocessed_keypoints.append(-1)
                preprocessed_keypoints.append(-1)
            else:
                preprocessed_keypoints.append(point[0])
                preprocessed_keypoints.append(point[1])
        image_name_list.append(image_name)
        norm_keypoints_list.append(preprocessed_keypoints)
    return image_name_list, np.array(norm_keypoints_list)

def get_cluster_center_by_choice(cluster_centers, choice):
    filtered_centers = []
    for label in choice:
        filtered_centers.append(cluster_centers[label])
    return np.array(filtered_centers)

def write_cluster_centers(file_path, cluster_center_list):
    with open(file_path, 'a') as f:
        for cluster_center in cluster_center_list:
            for point in cluster_center:
                f.writelines(f'{point} ')
            f.writelines(f'{-1} {-1} ')
            f.writelines('\n')
    return

def write_pose_data(file_path, image_name_list, pose_keypoints_list, label_list):
    with open(file_path, 'a') as f:
        for idx in range(len(image_name_list)):
            image_name = image_name_list[idx]
            pose_keypoints = pose_keypoints_list[idx]
            label = label_list[idx]
            f.writelines(f'{image_name} ')
            for point in pose_keypoints:
                f.writelines(f'{point} ')
            f.writelines(f'{-1} {-1} {label}\n')
    return

def write_cluster_cnt(file_path, labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    cluster_counts = dict(zip(unique_labels, counts))

    for label, count in cluster_counts.items():
        with open(file_path, 'a') as f:
            f.writelines(f"Cluster {label}: {count}\n")