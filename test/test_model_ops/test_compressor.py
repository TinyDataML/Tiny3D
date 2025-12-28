import os
import sys
import torch
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 2)[0]  
sys.path.append(config_path)
from deephub.detection_model import Pointpillars, Centerpoint
from mmcv.runner import load_checkpoint
from model.model_deployor.deployor_utils import create_input
from nni.compression.pytorch.pruning import L1NormPruner
from model.model_deployor.deployor import deploy
from nni.compression.pytorch.speedup import ModelSpeedup
from model.model_compressor.compressor import *

import time
import faulthandler;faulthandler.enable()
import numpy as np
import copy
import torch.nn.utils.prune as prune

import unittest

class TestCompressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        input_names = ['voxels', 'num_points', 'coors']
        output_names = ['scores', 'bbox_preds', 'dir_scores']
        dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}

        pcd = '../../test/test_model_ops/data/kitti/kitti_000008.bin'
        checkpoint = '../../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        dataset = 'kitti'
        model_name = 'pointpillars'
        # device = 'cpu'
        backend = 'onnxruntime'
        output = 'pointpillars'
        fp16 = False
        model = Pointpillars()
        load_checkpoint(model, '../../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth', map_location='cpu')
        model.eval()
        cls.model = model

    # noinspection DuplicatedCode
    def test_static_quant(self):
        device = 'cpu'
        model.cpu()

        data, model_inputs = create_input(pcd, dataset, model_name, device)

        backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                            backend=backend, output_file=output, fp16=fp16, dataset=dataset)

        input_data = [model_inputs[0], model_inputs[1], model_inputs[2]]

        model_int8 = static_quant(model, input_data)

        torch_out = model_int8(model_inputs[0], model_inputs[1], model_inputs[2])

    def test_dynamic_quant(self):
        device = 'cuda:0'
        model.cuda()

        data, model_inputs = create_input(pcd, dataset, model_name, device)

        backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                            backend=backend, output_file=output, fp16=fp16, dataset=dataset)

        dynamic_quant(model)

        torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])

    
    def test_prune_1(self):
        device = 'cuda:0'
        model.cuda()

        data, model_inputs = create_input(pcd, dataset, model_name, device)

        backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                            backend=backend, output_file=output, fp16=fp16, dataset=dataset)

        prune_list = [torch.nn.Conv2d, torch.nn.Linear]
        amount_list = [0.3, 0.9]

        prune_1(model, prune_list, amount_list)

        torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])


if __name__ == '__main__':
    unittest.main()