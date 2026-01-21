# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from mmcv.parallel import collate, scatter
import numpy as np
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type
from typing import Dict, Sequence, Union
import importlib
if importlib.util.find_spec('tensorrt') is not None:
    import tensorrt as trt

import onnx
import torch
from packaging import version



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

trt_input_shapes = {
    'kitti': {
        'voxels': {
            'min_shape': [2000, 32, 4],
            'opt_shape': [5000, 32, 4],
            'max_shape': [9000, 32, 4]
        },
        'num_points': {
            'min_shape': [2000],
            'opt_shape': [5000],
            'max_shape': [9000]
        },
        'coors': {
            'min_shape': [2000, 4],
            'opt_shape': [5000, 4],
            'max_shape': [9000, 4]
        }
    },
    'nuscenes': {
        'voxels': {
            'min_shape': [5000, 20, 4],
            'opt_shape': [20000, 20, 4],
            'max_shape': [30000, 20, 4]
        },
        'num_points': {
            'min_shape': [5000],
            'opt_shape': [20000],
            'max_shape': [30000]
        },
        'coors': {
            'min_shape': [5000, 4],
            'opt_shape': [20000, 4],
            'max_shape': [30000, 4]
        }
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

def create_trt_engine(onnx_model: Union[str, onnx.ModelProto],
                      input_shapes: Dict[str, Sequence[int]],
                      log_level,
                      fp16_mode: bool = False,
                      int8_mode: bool = False,
                      int8_param: dict = None,
                      max_workspace_size: int = 0,
                      device_id: int = 0,
                      **kwargs):
    """Create a tensorrt engine from ONNX.

    Args:
        onnx_model (str or onnx.ModelProto): Input onnx model to convert from.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.INFO`.
        fp16_mode (bool): Specifying whether to enable fp16 mode.
            Defaults to `False`.
        int8_mode (bool): Specifying whether to enable int8 mode.
            Defaults to `False`.
        int8_param (dict): A dict of parameter  int8 mode. Defaults to `None`.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `0`.
        device_id (int): Choice the device to create engine. Defaults to `0`.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine created from onnx_model.

    Example:
        >>> engine = create_trt_engine(
        >>>             "onnx_model.onnx",
        >>>             {'input': {"min_shape" : [1, 3, 160, 160],
        >>>                        "opt_shape" : [1, 3, 320, 320],
        >>>                        "max_shape" : [1, 3, 640, 640]}},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
    """
    device = torch.device('cuda:{}'.format(device_id))
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 0:
        profile = builder.create_optimization_profile()

        for input_name, param in input_shapes.items():
            min_shape = param['min_shape']
            opt_shape = param['opt_shape']
            max_shape = param['max_shape']
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5Calibrator(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'
    return engine

def save_trt_engine(engine, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to be serialized.
        path (str): The absolute disk path to write the engine.
    """
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))

def load_trt_engine(path: str):
    """Deserialize TensorRT engine from disk.

    Args:
        path (str): The disk path to read the engine.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    """
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine

def torch_dtype_from_trt(dtype) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')

def torch_device_from_trt(device):
    """Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by torch')

def print_trt_engine(engine):
    """
    Print trt engine information.

    Args:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    Returns:
        None
    """
    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)

        print('input id:', idx, '   is input: ', is_input, '  binding name:', name, '  shape:', shape, 'type: ',
              op_type)