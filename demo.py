import os
import sys
import torch

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

    load_checkpoint(model, 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth', map_location='cpu')
    model.cpu()

    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.dtype)

    model.eval()
    # print(model)
    # print('--------------------------------------------------------')
    input_names = ['voxels', 'num_points', 'coors']
    output_names = ['scores', 'bbox_preds', 'dir_scores']
    dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}
    # dynamic_axes = None

    # #0 NNI pruning
    # ----------------------------------------
    # config_list = [{
    # 'sparsity_per_layer': 0.5,
    # 'op_types': ['Linear', 'Conv2d']
    # }]

    # pruner = L1NormPruner(model, config_list)

    # _, masks = pruner.compress()

    # for name, mask in masks.items():
    #     print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    # pruner._unwrap_model()

    # from nni.compression.pytorch.speedup import ModelSpeedup
    # ----------------------------------------


    pcd = 'test/test_model_ops/data/kitti/kitti_000008.bin'
    checkpoint = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    dataset = 'kitti'
    model_name = 'pointpillars'
    device = 'cpu'
    backend = 'onnxruntime'
    output = 'pointpillars'
    fp16 = False

    data, model_inputs = create_input(pcd, dataset, model_name, device)

    backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                          backend=backend, output_file=output, fp16=fp16, dataset=dataset)
    print("SIZESIZESIZE : ", np.array(model_inputs[0].cpu()).shape)
    print("SIZESIZESIZE : ", np.array(model_inputs[1].cpu()).shape)
    print("SIZESIZESIZE : ", np.array(model_inputs[2].cpu()).shape)


    # #0 NNI pruning
    #----------------------------------------
    # ModelSpeedup(model, [model_inputs[0], model_inputs[1].short(), model_inputs[2].short()], masks).speedup_model()
    #----------------------------------------

    # #1 dynamic quant (torch)
    #----------------------------------------
    # model.cpu()
    # torch.quantization.quantize_dynamic(model, {torch.nn.Linear},  dtype=torch.qint8)
    # model.cuda()
    # dynamic_quant(model)
    #----------------------------------------

    # # #2 static quant (torch)
    # #----------------------------------------
    # # print(model)
    input_data = [model_inputs[0], model_inputs[1], model_inputs[2]]

    static_quant(model, input_data)
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # model_prepared = torch.quantization.prepare(model)

    # model_prepared(model_inputs[0], model_inputs[1], model_inputs[2])
    # # model_prepared.cpu()


    # model_prepared.cpu()
    # model_int8 = torch.quantization.convert(model_prepared, inplace=True)

    # # print(model_int8)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # # model_int8.cpu()
    # # model_int8.cuda()
    # # data, model_inputs_cuda = create_input(pcd, dataset, model_name, 'cuda:0')
    # torch_out = model_int8(model_inputs[0], model_inputs[1], model_inputs[2])
    #----------------------------------------

    # for n,p in model.named_parameters():
	#     print(n)
	#     print(p)

    # torch_prune
    #----------------------------------------
    # prune_list = [torch.nn.Conv2d, torch.nn.Linear]
    # amount_list = [0.3, 0.9]

    # torch_prune(model, prune_list, amount_list)
    #----------------------------------------
    
    # for n,p in model.named_parameters():
	#     print(n)
	#     print(p)


    # torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])


    # if backend == 'onnxruntime':
    #     import onnxruntime

    #     ort_session = onnxruntime.InferenceSession(backend_file)

    #     input_dict = {}
    #     input_dict['voxels'] = model_inputs[0].cpu().numpy()
    #     input_dict['num_points'] = model_inputs[1].cpu().numpy()
    #     input_dict['coors'] = model_inputs[2].cpu().numpy()
    #     ort_output = ort_session.run(['scores', 'bbox_preds', 'dir_scores'], input_dict)

    #     outputs = {}
    #     outputs['scores'] = torch.tensor(ort_output[0])
    #     outputs['bbox_preds'] = torch.tensor(ort_output[1])
    #     outputs['dir_scores'] = torch.tensor(ort_output[2])

    #     print('onnx : inference successful!')
    
    print(time.time() - start)

if __name__ == '__main__':
    main()