# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from mmcv.parallel import collate, scatter
import numpy as np
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

test_pipelines = {
    'pointpillars': {
        'kitti': [
                   dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                   dict(
                        type='MultiScaleFlipAug3D',
                        img_scale=(1333, 800),
                        pts_scale_ratio=1,
                        flip=False,
                        transforms=[
                            dict(
                                type='GlobalRotScaleTrans',
                                rot_range=[0, 0],
                                scale_ratio_range=[1.0, 1.0],
                                translation_std=[0, 0, 0]),
                            dict(type='RandomFlip3D'),
                            dict(
                                type='PointsRangeFilter',
                                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                            dict(
                                type='DefaultFormatBundle3D',
                                class_names=['Car'],
                                with_label=False),
                            dict(type='Collect3D', keys=['points'])
                        ])
                ],
        'waymo': [
                   dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                   dict(
                        type='MultiScaleFlipAug3D',
                        img_scale=(1333, 800),
                        pts_scale_ratio=1,
                        flip=False,
                        transforms=[
                            dict(
                                type='GlobalRotScaleTrans',
                                rot_range=[0, 0],
                                scale_ratio_range=[1.0, 1.0],
                                translation_std=[0, 0, 0]),
                            dict(type='RandomFlip3D'),
                            dict(
                                type='PointsRangeFilter',
                                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                            dict(
                                type='DefaultFormatBundle3D',
                                class_names=['Car'],
                                with_label=False),
                            dict(type='Collect3D', keys=['points'])
                        ])
                ],
    },
    'centerpoint': {
        'nuscenes': [
                   dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=4,
                        file_client_args=dict(backend='disk')),
                   dict(
                        type='MultiScaleFlipAug3D',
                        img_scale=(1333, 800),
                        pts_scale_ratio=1,
                        flip=False,
                        transforms=[
                            dict(
                                 type='GlobalRotScaleTrans',
                                 rot_range=[0, 0],
                                 scale_ratio_range=[1.0, 1.0],
                                 translation_std=[0, 0, 0]),
                            dict(type='RandomFlip3D'),
                            dict(
                                 type='DefaultFormatBundle3D',
                                 class_names=[
                                             'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                             'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                                             'traffic_cone'
                                             ],
                                 with_label=False),
                            dict(type='Collect3D', keys=['points'])
        ])
],
    }
}


voxel_layers = {
    'pointpillars': {
        'kitti': dict(
                        max_num_points=32,
                        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                        voxel_size=[0.16, 0.16, 4],
                        max_voxels=(16000, 40000)),
        'waymo': dict(
                        max_num_points=10,
                        point_cloud_range=[-76.8, -51.2, -2, 76.8, 51.2, 4],
                        voxel_size=[0.08, 0.08, 0.1],
                        max_voxels=(80000, 90000))
    },
    'centerpoint': {
        'nuscenes': dict(
                        max_num_points=20,
                        voxel_size=[0.2, 0.2, 8],
                        max_voxels=(30000, 40000),
                        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    }
}

def create_input(pcd, dataset, model, device):
    """Create input for detector.

    Args:
        pcd (str): Input pcd file path.

    Returns:
        tuple: (data, input), meta information for the input pcd
            and model input.
    """
    data = read_pcd_file(pcd, test_pipelines[model][dataset], device, box_type_3d='LiDAR')
    voxels, num_points, coors = voxelize(
        voxel_layers[model][dataset], data['points'][0])
    return data, (voxels, num_points, coors)

def read_pcd_file(pcd, test_pipeline, device, box_type_3d):
    """Read data from pcd file and run test pipeline.

    Args:
        pcd (str): Pcd file path.
        device (str): A string specifying device type.

    Returns:
        dict: meta information for the input pcd.
    """
    if isinstance(pcd, (list, tuple)):
        pcd = pcd[0]
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(
        box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [
        img_metas.data[0] for img_metas in data['img_metas']
    ]
    data['points'] = [point.data[0] for point in data['points']]
    if device != 'cpu':
        data = scatter(data, [device])[0]
    return data

def voxelize(voxel_layer, points):
    """convert kitti points(N, >=3) to voxels.

    Args:
        model_cfg (str | mmcv.Config): The model config.
        points (torch.Tensor): [N, ndim] float tensor. points[:, :3]
            contain xyz points and points[:, 3:] contain other information
            like reflectivity.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points
            and returned when max_points != -1.
        coordinates: [M, 3] int32 tensor, always returned.
        num_points_per_voxel: [M] int32 tensor. Only returned when
            max_points != -1.
    """
    from mmcv.ops import Voxelization
    voxel_layer = Voxelization(**voxel_layer)
    voxels, coors, num_points = [], [], []
    for res in points:
        res_voxels, res_coors, res_num_points = voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
    voxels = torch.cat(voxels, dim=0)
    num_points = torch.cat(num_points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0)
    return voxels, num_points, coors_batch