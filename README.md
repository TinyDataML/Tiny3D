# Tiny3D
[Tiny3D website](https://github.com/TinyDataML/Tiny3D)

[Documentation](https://github.com/TinyDataML/Tiny3D), [Tutorials and examples](https://github.com/TinyDataML/Tiny3D)

## Introduction 
Tiny3D is a light weight 3d object detection service production system.

## Features
Tiny3D solution embodies three transformative features: 
1. **A Performance Optimization Engine** for 3d object detection online/offline inference services product performance optimization. Through this engine users can easily get a high accuracy and high speed 3d object detection service/competetion result in a **Data-Centeric AI** way. Our Performance Optimization Engine can easily be a Plug-in to any machine learning system.
2. **One line of code** to complete dataset editing, model training, model testing, model compression, model deployment.
3. **A user-friendly web interface** for a developer team to product a 3d object detection service pictorially, in a low-code fashion. [currently not supported]

## Example
### step-1: Edit the data using different data operation method to get high quality dataset
```
from tiny3d.data import dataset_edit

dataset_edit(dataset_input_path, dataset_output_path, denoise_method=None, 
             simulation_method='Snow', filter_method=None, 
             augmentation_method=None, qualification_method=None)
```

### step-2: Train a model on edited dataset
```
from tiny3d.deephub import Pointpillars
from tiny3d.engine import Pointpillars_engine, fit, build_dataset

torch_model = Pointpillars()
model = Pointpillars_engine(torch_model)

dataset_train = build_dataset(train_dataset_path)
dataset_val = build_datasetvcal_dataset_path)

fit(dataset_train=dataset_train, dataset_val=dataset_val, torch_model=model)
```

### step-3: Compress a trained model
```
from tiny3d.model.model_compressor import torch_prune, dynamic_quant 
prune_list = [torch.nn.Conv2d, torch.nn.Linear]
amount_list = [0.3, 0.9]
torch_prune(model, prune_list, amount_list)

dynamic_quant(model)
```
### step-4: Deploy a model
```
from tiny3d.model.model_deployor import deploy 

backend='tensorrt'
backend_file = deploy(model, backend='tensorrt', output_file=output_model_path)
```
### step-5: Provide a model serving
```
from tiny3d.model.model_server import payload

PyTorch_REST_API_URL = 'http://127.0.0.1:1234/'
url = PyTorch_REST_API_URL + 'transfer'

# Submit the request
requests.post(url, files=payload).json()
```

## Lidar data operations currently supported
- lidar data loading
- lidar data sampling
- lidar data preprocessing/cleaning
- lidar data denoising
- lidar data outlier detection
- lidar data augmentation
- lidar data simulation

## Lidar based 3d object detection model operations currently supported
- model compression
- model deploy and serve
- model ensemble

## Data-Model co-operations currently supported
- training
- testing
- bad case visulization

## TODO
### 1. Add more data ops
- lidar data selection
- lidar data robustion
- lidar data privacy
- lidar data domain adptation
- lidar data auto-labeling
- data drift emergency
### 2. Add visual interaction interface.

## Acknowlegement
- [MLSys and MLOps Community](https://github.com/MLSysOps)
