# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from argparse import ArgumentParser

from deephub.detection_model import Pointpillars, Centerpoint

from mmcv.runner import load_checkpoint
from model.model_deployor.deployor import deploy
from model.model_deployor.deployor_utils import create_input
import onnx
import onnxruntime
import time
import importlib
if importlib.util.find_spec('tensorrt') is not None:
    from model.model_deployor.onnx2tensorrt import load_trt_engine, torch_dtype_from_trt, torch_device_from_trt
else:
    print('Please install TensorRT if you want to convert')

# --------------------------flask------------------------------------------

import io
import json

import flask
import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = True


@app.route("/transfer", methods=["POST"])
def transfer():
    # Initialize the data dictionary that will be returned from the view.
    if flask.request.method == 'POST':

        pcd=flask.request.files["pcd"].read().decode("utf-8")
        checkpoint = flask.request.files["checkpoint"].read().decode("utf-8")
        backend = flask.request.files["backend"].read().decode("utf-8")
        output = flask.request.files["output"].read().decode("utf-8")
        dataset = flask.request.files["dataset"].read().decode("utf-8")
        model_type = flask.request.files["model_type"].read().decode("utf-8")
        device = flask.request.files["device"].read().decode("utf-8")

        # Init model and load checkpoints
        if model_type == 'pointpillars':
            model = Pointpillars()
        elif model_type == 'centerpoint':
            model = Centerpoint()

        load_checkpoint(model, checkpoint, map_location='cpu')
        model.cuda()
        # model.cpu()
        model.eval()

        # define deploy params
        input_names = ['voxels', 'num_points', 'coors']
        output_names = ['scores', 'bbox_preds', 'dir_scores']
        dynamic_axes = {'voxels': {0: 'voxels_num'},
                        'num_points': {0: 'voxels_num'},
                        'coors': {0: 'voxels_num'}}
        # dynamic_axes = None
        fp16 = False

        data, model_inputs = create_input(pcd, dataset, model_type, device)

        # deploy
        backend_file = deploy(model, model_inputs, input_names, output_names, dynamic_axes,
                              backend=backend, output_file=output, fp16=fp16, dataset=dataset)

        # verify
        torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])

        result={"status" : False}

        if backend == 'onnxruntime':

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

            result["status"] = True
            result["model_path"] = output+backend

            return flask.jsonify(result)

        if backend == 'torchscript':
            jit_model = torch.jit.load(backend_file)
            script_output = jit_model(model_inputs[0], model_inputs[1], model_inputs[2])
            outputs = {}
            outputs['scores'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])
            outputs['bbox_preds'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])
            outputs['dir_scores'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])

            print("torchscript inference successful")

            result["status"] = True
            result["model_path"] = output+backend

            return flask.jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    if flask.request.method == 'POST':

        pcd=flask.request.files["pcd"].read().decode("utf-8")
        checkpoint = flask.request.files["checkpoint"].read().decode("utf-8")
        backend = flask.request.files["backend"].read().decode("utf-8")
        output = flask.request.files["output"].read().decode("utf-8")
        dataset = flask.request.files["dataset"].read().decode("utf-8")
        model_type = flask.request.files["model_type"].read().decode("utf-8")
        device = flask.request.files["device"].read().decode("utf-8")

        # Init model and load checkpoints
        if model_type == 'pointpillars':
            model = Pointpillars()
        elif model_type == 'centerpoint':
            model = Centerpoint()

        load_checkpoint(model, checkpoint, map_location='cpu')
        model.cuda()
        # model.cpu()
        model.eval()

        # define deploy params
        input_names = ['voxels', 'num_points', 'coors']
        output_names = ['scores', 'bbox_preds', 'dir_scores']
        dynamic_axes = {'voxels': {0: 'voxels_num'},
                        'num_points': {0: 'voxels_num'},
                        'coors': {0: 'voxels_num'}}
        # dynamic_axes = None
        fp16 = False

        data, model_inputs = create_input(pcd, dataset, model_type, device)

        # verify
        torch_out = model(model_inputs[0], model_inputs[1], model_inputs[2])

        result={"status" : False}

        if backend == 'onnxruntime':

            backend_file=output+".onnx"

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

            result["status"] = True
            result['scores'] = ort_output[0].tolist()
            result['bbox_preds'] = ort_output[1].tolist()
            result['coors'] = ort_output[2].tolist()

            return flask.jsonify(result)

        if backend == 'torchscript':

            backend_file=output+".jit"
            jit_model = torch.jit.load(backend_file)
            script_output = jit_model(model_inputs[0], model_inputs[1], model_inputs[2])
            outputs = {}
            outputs['scores'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])
            outputs['bbox_preds'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])
            outputs['dir_scores'] = torch.tensor([item.cpu().detach().numpy() for item in script_output[0]])

            print("torchscript inference successful")

            result["status"] = True
            result['scores'] = ort_output[0].tolist()
            result['bbox_preds'] = ort_output[1].tolist()
            result['coors'] = ort_output[2].tolist()

            return flask.jsonify(result)



if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ... \nPlease wait until server has fully started")
    app.run(port=1234)
