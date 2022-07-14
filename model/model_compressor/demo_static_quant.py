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
def main():
    start = time.time()
    model = Pointpillars()

    load_checkpoint(model, '../../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth', map_location='cpu')
    model.cpu()

    model.eval()

    input_names = ['voxels', 'num_points', 'coors']
    output_names = ['scores', 'bbox_preds', 'dir_scores']
    dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}
    # dynamic_axes = None

    pcd = '../../test/test_model_ops/data/kitti/kitti_000008.bin'
    checkpoint = '../../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    dataset = 'kitti'
    model_name = 'pointpillars'
    device = 'cpu'
    backend = 'onnxruntime'
    output = 'pointpillars'
    fp16 = False

    data, model_inputs = create_input(pcd, dataset, model_name, device)

    backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                          backend=backend, output_file=output, fp16=fp16, dataset=dataset)

    #2 static quant (torch)
    #----------------------------------------

    input_data = [model_inputs[0], model_inputs[1], model_inputs[2]]

    model_int8 = static_quant(model, input_data)

    torch_out = model_int8(model_inputs[0], model_inputs[1], model_inputs[2])
    #----------------------------------------


    # torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])


    if backend == 'onnxruntime':
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(backend_file)

        input_dict = {}
        input_dict['voxels'] = model_inputs[0].cpu().numpy()
        input_dict['num_points'] = model_inputs[1].cpu().numpy()
        input_dict['coors'] = model_inputs[2].cpu().numpy()
        ort_output = ort_session.run(['scores', 'bbox_preds', 'dir_scores'], input_dict)

        outputs = {}
        outputs['scores'] = torch.tensor(ort_output[0])
        outputs['bbox_preds'] = torch.tensor(ort_output[1])
        outputs['dir_scores'] = torch.tensor(ort_output[2])

        print('onnx : inference successful!')
    
    print(time.time() - start)

if __name__ == '__main__':
    main()