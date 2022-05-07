import torch

def deploy(model, model_inputs, input_names, output_names, dynamic_axes, backend='onnx', output_file='end2end', verbose=False):
    if backend == 'onnx':
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
        return output_file