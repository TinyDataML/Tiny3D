import torch
import unittest

import numpy as np
import onnxruntime
from deephub.detection_model import Pointpillars
from model.model_deployor.deployor import deploy
from model.model_deployor.deployor_utils import create_input_pointpillars
from model.model_deployor.onnx2tensorrt import load_trt_engine, torch_dtype_from_trt, torch_device_from_trt

pcd = 'test/data_tobe_tested/kitti/kitti_000008.bin'
device = 'cuda:0'
input_names = ['voxels', 'num_points', 'coors']
output_names = ['scores', 'bbox_preds', 'dir_scores']
dynamic_axes = {'voxels': {0: 'voxels_num'},
                    'num_points': {0: 'voxels_num'},
                    'coors': {0: 'voxels_num'}}
# dynamic_axes = None
fp16 = False


class TestModelDeployor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data, model_inputs = create_input_pointpillars(pcd, 'kitti', device)
        cls.model_inputs = model_inputs
        model = Pointpillars()
        model.cuda()
        model.eval()
        cls.model = model

    # noinspection DuplicatedCode
    def test_deployor_onnx(self):
        # Compute Pytorch model outputs
        torch_out = self.model(self.model_inputs[0], self.model_inputs[1], self.model_inputs[2])
        # deploy ONNX
        backend_file = deploy(self.model, self.model_inputs, input_names, output_names, dynamic_axes, backend='onnxruntime',
                              output_file='end2end', fp16=fp16)

        # Compute ONNX model outputs
        ort_session = onnxruntime.InferenceSession(backend_file)

        input_dict = {}
        input_dict['voxels'] = self.model_inputs[0].cpu().numpy()
        input_dict['num_points'] = self.model_inputs[1].cpu().numpy()
        input_dict['coors'] = self.model_inputs[2].cpu().numpy()
        ort_output = ort_session.run(['scores', 'bbox_preds', 'dir_scores'], input_dict)

        outputs = {}
        outputs['scores'] = torch.tensor(ort_output[0])
        outputs['bbox_preds'] = torch.tensor(ort_output[1])
        outputs['dir_scores'] = torch.tensor(ort_output[2])

        # test
        self.assertEqual(np.testing.assert_allclose(outputs['scores'].numpy().flatten(),
                                                    torch_out[0][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)
        self.assertEqual(np.testing.assert_allclose(outputs['bbox_preds'].numpy().flatten(),
                                                    torch_out[1][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)
        self.assertEqual(np.testing.assert_allclose(outputs['dir_scores'].numpy().flatten(),
                                                    torch_out[2][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)

    def test_deployor_trt(self):
        # Compute Pytorch model outputs
        torch_out = self.model(self.model_inputs[0], self.model_inputs[1], self.model_inputs[2])
        # deploy TensorRT
        backend_file = deploy(self.model, self.model_inputs, input_names, output_names, dynamic_axes,
                              backend='tensorrt',
                              output_file='end2end', fp16=fp16)

        # Compute TensorRT model outputs
        engine = load_trt_engine(backend_file)
        context = engine.create_execution_context()
        names = [_ for _ in engine]
        input_names_trt = list(filter(engine.binding_is_input, names))
        output_names_trt = list(set(names) - set(input_names_trt))
        input_dict = {
            'voxels': self.model_inputs[0],
            'num_points': self.model_inputs[1],
            'coors': self.model_inputs[2]
        }
        bindings = [None] * (len(input_names_trt) + len(output_names_trt))

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
        for output_name in output_names_trt:
            idx = engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
            shape = tuple(context.get_binding_shape(idx))

            device = torch_device_from_trt(engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        # test
        self.assertEqual(np.testing.assert_allclose(outputs['scores'].cpu().numpy().flatten(),
                                                    torch_out[0][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)
        self.assertEqual(np.testing.assert_allclose(outputs['bbox_preds'].cpu().numpy().flatten(),
                                                    torch_out[1][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)
        self.assertEqual(np.testing.assert_allclose(outputs['dir_scores'].cpu().numpy().flatten(),
                                                    torch_out[2][0].cpu().detach().numpy().flatten(), rtol=1e-03,
                                                    atol=1e-05), None)



if __name__ == '__main__':
    unittest.main()