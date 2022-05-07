# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from argparse import ArgumentParser
# import sys
# sys.path.append("/media/pc/sda/xwc2/pythonproject/Tiny3D/")
# print(sys.path)
from deephub.detection_model import Pointpillars

from mmcv.runner import load_checkpoint
from model.model_deployor.deployor import deploy
from model.model_deployor.deployor_utils import create_input_pointpillars



def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('backend', default='onnx', help='backend name')
    parser.add_argument('output', default='onnx', help='backend name')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    # Init model and load checkpoints
    model = Pointpillars()
    model.cuda()
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # define deploy params
    input_names = ['voxels', 'num_points', 'coors']
    output_names = ['scores', 'bbox_preds', 'dir_scores']
    dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}

    data, model_inputs = create_input_pointpillars(args.pcd, 'kitti', args.device)

    # deploy
    backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes, backend=args.backend, output_file=args.output)

    # verify
    if args.backend == 'onnx':
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

        print('Successful!')




if __name__ == '__main__':
    main()
