import numpy as np

from qualificator_utils import find_label_issues

def lidar_qualificator(lidar_dataaet, pred, method):
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
    points = lidar_dataset['points']
    labels = lidar_dataaet['gt_label']
    # pred_bbox = lidar_data['pred_bbox']
    # gt_bbox = lidar_data['gt_bbox']
    points = points.numpy()

    if method == 'confident_learning':
        points, pred_bbox, gt_bbox, issues = find_label_issues(points, labelsm pred)

    lidar_data['points'] = points
    lidar_data['pred_bbox'] = pred_bbox
    lidar_data['gt_bbox'] = gt_bbox
    return lidar_data, issues
