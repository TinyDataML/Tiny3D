import numpy as np
# import mayavi.mlab
from os import path as osp

  
import numpy as np

from .visualization_utils import mkdir_or_exist, _write_obj, _write_oriented_bbox


def lidar_visualizor(lidar_data,                            
                          out_dir='./',
                          filename='test',
                          method='open3d',
                          show=False,
                          snapshot=False,
                          pred_labels=None):
    """
        Use different visualize methods to visualize lidar data.
        Args:
            points: np.array
            method: str
        return:
            0
        Reference:
            https://github.com/strawlab/python-pcl
    """
    
    points = lidar_data['points']
    pred_bboxes = lidar_data['pred_bboxes']
    gt_bboxes = lidar_data['gt_bboxes']
    if method == 'open3d':
        result_path = osp.join(out_dir, filename)
        mkdir_or_exist(result_path)
    
        if show:
            from .open3d_vis import Visualizer
    
            vis = Visualizer(points)
            if pred_bboxes is not None:
                if pred_labels is None:
                    vis.add_bboxes(bbox3d=pred_bboxes)
                else:
                    palette = np.random.randint(
                        0, 255, size=(pred_labels.max() + 1, 3)) / 256
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(pred_bboxes[j])
                    for i in labelDict:
                        vis.add_bboxes(
                            bbox3d=np.array(labelDict[i]),
                            bbox_color=palette[i],
                            points_in_box_color=palette[i])
    
            if gt_bboxes is not None:
                vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
            show_path = osp.join(result_path,
                                 f'{filename}_online.png') if snapshot else None
            vis.show(show_path)
    
        if points is not None:
            _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))
    
        if gt_bboxes is not None:
            # bottom center to gravity center
            gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
    
            _write_oriented_bbox(gt_bboxes,
                                 osp.join(result_path, f'{filename}_gt.obj'))
    
        if pred_bboxes is not None:
            # bottom center to gravity center
            pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
    
            _write_oriented_bbox(pred_bboxes,
                                 osp.join(result_path, f'{filename}_pred.obj'))

    return 0


