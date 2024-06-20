import os
import cv2
import random

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

def vis_pose(image_path, pose_keypoints, mask_box=None):
    image = cv2.imread(image_path)
    print(image.shape)
    print(mask_box)
    if mask_box != None:
        image[mask_box[0]:mask_box[2], mask_box[1]:mask_box[3]] = 0

    if pose_keypoints[5][0] > 0 and pose_keypoints[6][0] > 0:
        neck = ((pose_keypoints[5][0] + pose_keypoints[6][0]) // 2, (pose_keypoints[5][1] + pose_keypoints[6][1]) // 2)
    else:
        neck = (-1, -1)
    pose_keypoints.append(neck)

    pose_keypoints = [(x[1], x[0]) for x in pose_keypoints]
    
    for idx, pair in enumerate(link_pairs):
        if pose_keypoints[pair[0]][0] == -1 or pose_keypoints[pair[1]][0] == -1:
            continue
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], link_color, 2)
    
    for idx, point in enumerate(pose_keypoints):
        if point[0] == -1:
            continue
        cv2.circle(image, point, 5, point_color[idx], thickness=-1)
        cv2.putText(image, f'{idx}', org=(point[0] - 5, point[1] - 5), fontFace=0, fontScale=0.5, color=point_color[idx], thickness=2)
    # image = cv2.resize(image, (512, 512))
    cv2.imshow(image_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return