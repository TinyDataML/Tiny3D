import numpy as np

from qualificator_utils import find_label_issues

def lidar_qualificator(lidar_data, method):
    """
        Use different Data quality assessment methods to qualificate lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            simulated lidar_data
        Reference:
            https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py

    """
    points = lidar_data['points']
    pred_bbox = lidar_data['pred_bbox']
    gt_bbox = lidar_data['gt_bbox']
    points = points.numpy()

    if method == 'confident_learning':
        points, pred_bbox, gt_bbox, issues = find_label_issues(points, pred_bbox, gt_bbox)    

    lidar_data['points'] = points
    lidar_data['pred_bbox'] = pred_bbox
    lidar_data['gt_bbox'] = gt_bbox
    return lidar_data, issues
