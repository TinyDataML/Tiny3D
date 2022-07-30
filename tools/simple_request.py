# coding=utf-8

import requests
import argparse
import json

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:1234/'

def model_transfer(args):

    url = PyTorch_REST_API_URL + 'transfer'
    payload={"pcd" : args.pcd,
             "checkpoint" : args.checkpoint,
             "backend" : args.backend,
             "output" : args.output,
             "dataset" : args.dataset,
             "model_type" : args.model_type,
             "device" : args.device}

    print("\nTransfering ... please wait\n")

    # Submit the request
    r = requests.post(url, files=payload).json()

    if r["status"]:
        if args.backend == "onnxruntime":
            print("Model transfer success")
            print("ONNX model of ", args.model_type, " based on ", args.dataset, "dataset is saved as: ", args.output+".onnx")
        if args.backend == "torchscript":
            print("Model transfer success")
            print("TorchScript model of ", args.model_type, " based on ", args.dataset, "dataset is saved as: ", args.output+".jit")
    else:
        print("request failed")


def model_predict(args):

    url = PyTorch_REST_API_URL + 'predict'
    payload={"pcd" : args.pcd,
             "checkpoint" : args.checkpoint,
             "backend" : args.backend,
             "output" : args.output,
             "dataset" : args.dataset,
             "model_type" : args.model_type,
             "device" : args.device}

    print("\nComputing ... please wait\n")

    # Submit the request
    r = requests.post(url, files=payload).json()

    if r["status"]:
        print("predict success, the result is saved as: ")
        b=json.dumps(r)
        f2=open("result.json", 'w')
        f2.write(b)
        f2.close()
        print("result.json")
    else:
        print("request failed")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--type', help='model transfer or model predict, support: transfer, predict')
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--backend', default='onnxruntime', help='support: onnxruntime, torchscript, tensorrt')
    parser.add_argument('--output', default='onnxModel', help='output model file name')
    parser.add_argument('--dataset', default='kitti', help='support: kitti, nuscenes')
    parser.add_argument('--model_type', default='pointpillars', help='support: pointpillars, centerpoint')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args=parser.parse_args()

    if args.type == "transfer":
        model_transfer(args)
    elif args.type == "predict":
        model_predict(args)

