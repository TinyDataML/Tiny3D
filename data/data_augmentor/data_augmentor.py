from augmentation_utils import *

def lidar_augmentation(lidar_data, method):
    """
        Use different data augmentation methods to augment lidar data.

        Args:
            lidar_data: dict
            method: str
        return:
            augmentated lidar_data

        Reference:
            https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/augmentor/data_augmentor.py
    """
    gt_boxes = lidar_data['gt_boxes']
    points = lidar_data['points']

    if method == 'random_flip_along_x':
        gt_boxes, points = random_flip_along_x(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_z':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)

    lidar_data['gt_boxes'] = gt_boxes
    lidar_data['points'] = points

    return lidar_data
from augmentation_utils import *

def lidar_augmentation(lidar_data, method):
    """
        Use different data augmentation methods to augment lidar data.

        Args:
            lidar_data: dict
            method: str
        return:
            augmentated lidar_data

        Reference:
            https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/augmentor/data_augmentor.py
    """
    gt_boxes = lidar_data['gt_boxes']
    points = lidar_data['points']

    if method == 'random_flip_along_x':
        gt_boxes, points = random_flip_along_x(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_z':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)
    elif method == 'random_flip_along_y':
        gt_boxes, points = random_flip_along_y(gt_boxes, points)

    lidar_data['gt_boxes'] = gt_boxes
    lidar_data['points'] = points

    return lidar_data
