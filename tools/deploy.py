# Copyright (c) OpenMMLab. All rights reserved.

import torch
from argparse import ArgumentParser

from deephub.detection_model import Pointpillars, Centerpoint

from mmcv.runner import load_checkpoint
from model.model_deployor.deployor import deploy
from model.model_deployor.deployor_utils import create_input
import importlib
if importlib.util.find_spec('tensorrt') is not None:
    from model.model_deployor.onnx2tensorrt import load_trt_engine, torch_dtype_from_trt, torch_device_from_trt
else:
    print('Please install TensorRT if you want to convert')


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('backend', default='onnx', help='backend name')
    parser.add_argument('output', default='onnx', help='backend name')
    parser.add_argument('dataset', default='onnx', help='backend name')
    parser.add_argument('model_name', default='onnx', help='backend name')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    # Init model and load checkpoints
    if args.model_name == 'pointpillars':
        model = Pointpillars()
    elif args.model_name == 'centerpoint':
        model = Centerpoint()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    # define deploy params
    input_names = ['voxels', 'num_points', 'coors']
    output_names = ['scores', 'bbox_preds', 'dir_scores']
    dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}
    # dynamic_axes = None
    fp16 = False

    data, model_inputs = create_input(args.pcd, args.dataset, args.model_name, args.device)

    # deploy
    backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                          backend=args.backend, output_file=args.output, fp16=fp16, dataset=args.dataset)

    # verify
    torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])
    if args.backend == 'onnxruntime':
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

        print('inference successful!')

    if args.backend == 'tensorrt':
        engine = load_trt_engine(backend_file)
        context = engine.create_execution_context()
        names = [_ for _ in engine]
        input_names = list(filter(engine.binding_is_input, names))
        output_names = list(set(names) - set(input_names))
        input_dict = {
            'voxels': model_inputs[0],
            'num_points': model_inputs[1],
            'coors': model_inputs[2]
        }
        bindings = [None] * (len(input_names) + len(output_names))

        profile_id = 0
        for input_name, input_tensor in input_dict.items():
            # check if input shape is valid
            profile = engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in output_names:
            idx = engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
            shape = tuple(context.get_binding_shape(idx))

            device = torch_device_from_trt(engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        print('inference successful!')




if __name__ == '__main__':
    main()
