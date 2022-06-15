from .augmentation_utils import *

def lidar_augmentation(lidar_data, method, rot_range=[-0.78539816, 0.78539816], scale_range=[0.95, 1.05], intensity_range=[0, 0.2], offset_std=[-0.2, 0.2] ):
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
        
    elif method == 'global_rotation':
        gt_boxes, points = global_rotation(gt_boxes, points, rot_range)
    elif method == 'global_scaling':
        gt_boxes, points = global_scaling(gt_boxes, points, scale_range)
        
    elif method == 'random_translation_along_x':
        gt_boxes, points = random_translation_along_x(gt_boxes, points, offset_std)
    elif method == 'random_translation_along_y':
        gt_boxes, points = random_translation_along_y(gt_boxes, points, offset_std)
    elif method == 'random_translation_along_z':
        gt_boxes, points = random_translation_along_z(gt_boxes, points, offset_std)
        
    elif method == 'global_frustum_dropout_top':
        gt_boxes, points = global_frustum_dropout_top(gt_boxes, points, intensity_range)
    elif method == 'global_frustum_dropout_bottom':
        gt_boxes, points = global_frustum_dropout_bottom(gt_boxes, points, intensity_range)
    elif method == 'global_frustum_dropout_left':
        gt_boxes, points = global_frustum_dropout_left(gt_boxes, points, intensity_range)
    elif method == 'global_frustum_dropout_right':
        gt_boxes, points = global_frustum_dropout_right(gt_boxes, points, intensity_range)
        
    elif method == 'local_scaling':
        gt_boxes, points = local_scaling(gt_boxes, points, scale_range)
    elif method == 'local_rotation':
        gt_boxes, points = local_rotation(gt_boxes, points, rot_range)
        
    elif method == 'random_local_translation_along_x':
        gt_boxes, points = random_local_translation_along_x(gt_boxes, points, offset_range)
    elif method == 'random_local_translation_along_y':
        gt_boxes, points = random_local_translation_along_y(gt_boxes, points, offset_range)
    elif method == 'random_local_translation_along_z':
        gt_boxes, points = random_local_translation_along_z(gt_boxes, points, offset_range)
        
    elif method == 'local_frustum_dropout_top':
        gt_boxes, points = local_frustum_dropout_top(gt_boxes, points, intensity_range)
    elif method == 'local_frustum_dropout_bottom':
        gt_boxes, points = local_frustum_dropout_bottom(gt_boxes, points, intensity_range)
    elif method == 'local_frustum_dropout_left':
        gt_boxes, points = local_frustum_dropout_left(gt_boxes, points, intensity_range)
    elif method == 'local_frustum_dropout_right':
        gt_boxes, points = local_frustum_dropout_right(gt_boxes, points, intensity_range)
    
    lidar_data['gt_boxes'] = gt_boxes
    lidar_data['points'] = points

    return lidar_data
