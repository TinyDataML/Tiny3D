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





# Demo

## deployor

```shell
python tools/deploy.py
/the/path/of/kitti_000008.bin
/the/path/of/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
onnxruntime
pointpillars
```























