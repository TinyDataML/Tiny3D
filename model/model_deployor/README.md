

# Deployor

We offer deployment of point cloud detection models pointpillars and centerpoint as ONNXRuntime and TensorRT.

The deployment process is as follows:

**1.Generate the inputs needed for the model based on the model's config (including reading point cloud files, doing data augmentation, voxelization.)**

```python
def create_input(pcd, dataset, model, device):
    """Create input for detector.

    Args:
        pcd (str): Input pcd file path.

    Returns:
        tuple: (data, input), meta information for the input pcd
            and model input.
    """
    data = read_pcd_file(pcd, test_pipelines[model][dataset], device, box_type_3d='LiDAR')
    voxels, num_points, coors = voxelize(
        voxel_layers[model][dataset], data['points'][0])
    return data, (voxels, num_points, coors)
```

**2.Rewrite the forward function of the network model to remove the judgments and operations that are not suitable for deployment**

**3.Convert Pytorch model to intermediate representation ONNX**

```python
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
```

**4.Convert the intermediate representation ONNX to trt if needed**

```python
engine = create_trt_engine(
            output_file,
            input_shapes=trt_input_shapes[dataset],
            fp16_mode=fp16,
            int8_mode=False,
            int8_param={},
            max_workspace_size=1073741824,
            device_id=0)
```



## Usage

You should first install onnx, onnxruntime as:

```shell
pip install onnx
pip install onnxruntime==1.8.1
```

You should first install tensorrt as:

1.download TensorRT from [NVIDIA Developer Program Membership Required | NVIDIA Developer](https://developer.nvidia.com/nvidia-tensorrt-download)

2.install tensorrt in python

```shell
cd /the/path/of/tensorrt/tar/gz/file
tar -zxvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp37-none-linux_x86_64.whl
```

3.add tensorrt into the environment variable

```shell
vim .bashrc
export LD_LIBRARY_PATH=../TensorRT-8.2.3.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=../TensorRT-8.2.3.0/lib:$LIBRARY_PATH
source .bashrc
```





You can then execute the following script to convert the model:

```shell
python tools/deploy.py
test/data_tobe_tested/kitti/kitti_000008.bin
checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
onnxruntime
pointpillars
kitti
pointpillars
```



```shell
python tools/deploy.py
test/data_tobe_tested/kitti/kitti_000008.bin
checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
tensorrt
pointpillars
kitti
pointpillars
```



