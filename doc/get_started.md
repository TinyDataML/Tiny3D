# Installation

This is the installation process on 3090 machine：

```shell
conda create --name tiny3d python=3.7
conda activate tiny3d
# install pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
# install mmdet
pip install mmdet
# install mmseg
pip install mmsegmentation
# install mmdet3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
# install other requirements
pip install -r requirements.txt
```



## deployor

### ONNX

```shell
pip install onnx
pip install onnxruntime==1.8.1
```



### TensorRT

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
export LD_LIBRARY_PATH=/home/xwc2/pythonproject/TensorRT-8.2.3.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/xwc2/pythonproject/TensorRT-8.2.3.0/lib:$LIBRARY_PATH
source .bashrc
```









Versions of some packages：

- mmcv-full = 1.5.0
- mmdet = 2.24.1
- mmsegmentation = 0.24.1
- mmdet3d = 1.0.0rc2
- onnx = 1.11.0
- onnxruntime = 1.8.1
- tensorrt = 8.2.3.0


## engine

### Pytorch Lighting

```shell
pip install pytorch-lightning
```


# Demo

## deployor

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
test/data_tobe_tested/nuscenes/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
checkpoints/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth
onnxruntime
centerpoint
nuscenes
centerpoint
```

checkpoints:
[pointpillars](https://drive.google.com/file/d/1YZoL6J9tGc43kgFlf9mBx8pw0fZrczjd/view?usp=sharing)
[centerpoint](https://drive.google.com/file/d/1kL-6ZUmamlMH06ADLkQ0Y2sPbh1JBZ1b/view?usp=sharing)


## engine

You shoule prepare Kitti dataset first as https://github.com/open-mmlab/mmdetection3d/blob/master/docs/zh_cn/datasets/kitti_det.md

The folder structure should be organized as follows.
```shell
Tiny3D
├── checkpoints
├── deephub
├── engine
├── test
├── tools
├── kitti
```

run
```shell
python tools/engines.py
--model_name
pointpillars
--mode
fit/eval/predict/inference/infer_production
--config
engine/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py
```



















