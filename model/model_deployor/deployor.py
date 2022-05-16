import torch
import tensorrt as trt
from model.model_deployor.onnx2tensorrt import create_trt_engine, save_trt_engine

trt_input_shapes_kitti = {
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
}


def deploy(model, model_inputs, input_names, output_names, dynamic_axes, backend='onnxruntime', output_file='end2end', verbose=False, fp16=False):
    """
        Deploy pytorch model to different backends.
        Args:
            model: torch.nn.module
            model_inputs: tensor
            input_names: deployment model input names
            output_names: deployment model output names
            dynamic_axes: specifies the dynamic dimension of the deployment model
            backend: specify convert backend
            output_file: output file name
            fp16: TensorRT fp16
        Return:
            backend file name
        Reference:
            https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py
    """
    assert backend in ['onnxruntime', 'tensorrt'], 'This backend isn\'t supported now!'

    output_file = output_file + '.onnx'
    torch.onnx.export(
        model,
        model_inputs,
        output_file,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=False,
        verbose=verbose)
    if backend == 'onnxruntime':
        return output_file
    if backend == 'tensorrt':
        engine = create_trt_engine(
            output_file,
            input_shapes=trt_input_shapes_kitti,
            log_level=trt.Logger.INFO,
            fp16_mode=fp16,
            int8_mode=False,
            int8_param={},
            max_workspace_size=1073741824,
            device_id=0)
        output_file = output_file.replace('onnx', 'trt')
        save_trt_engine(engine, output_file)
        return output_file