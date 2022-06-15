# [PointCleanNet](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/pointcleannet.html)
This is our implementation of PointCleannet, a network removes outliers and reduces noise in unordered point clouds.


![PointCleanNet cleans point clouds](https://raw.githubusercontent.com/mrakotosaon/pointcleannet/master/images/teaser.png "PointCleanNet")

The architecture is similar to [PCPNet](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/) (with a few smaller modifications).

This code was written by [Marie-Julie Rakotosaona](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/), based on the excellent implementation of PCPNet by [Paul Guerrero](https://paulguerrero.github.io) and [Yanir Kleiman](https://www.cs.tau.ac.il/~yanirk/).

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 2.7
* PyTorch 1.0

## Setup
Install required python packages, if they are not already installed ([tensorboardX](https://github.com/lanpa/tensorboard-pytorch) is only required for training):
``` bash
pip install numpy
pip install scipy
pip install tensorboardX
```


Clone this repository:
``` bash
git clone https://github.com/mrakotosaon/pointcleannet.git
cd pointcleannet
```


Download datasets:
``` bash
cd data
python download_data.py --task denoising
python download_data.py --task outliers_removal
```


Download pretrained models:
``` bash
cd models
python download_models.py --task denoising
python download_models.py --task outliers_removal
```

 ## Data

Our data can be found here: https://nuage.lix.polytechnique.fr/index.php/s/xSRrTNmtgqgeLGa .

It contains the following files:
- Dataset for denoising
- Training set and test set for outliers removal
- Pre-trained models for denoising and outliers removal

In the datasets the input and ground truth point clouds are stored in different files with the same name but with different extensions.
- For denoising: `.xyz` for input noisy point clouds, `.clean_xyz` for the ground truth.
- For outliers removal: `.xyz` for input point clouds with outliers, `.outliers` for the labels.



## Removing outliers
To classify outliers using default settings:
``` bash
cd outliers_removal
mkdir results
python eval_pcpnet.py
```

## Denoising
To denoise point clouds using default settings:
``` bash
cd noise_removal
mkdir results
./run.sh
```
(the input shapes and number of iterations are specified in run.sh file)


## Training
To train PCPNet with the default settings:
``` bash
python train_pcpnet.py
```

## Citation
If you use our work, please cite our paper.
```
@inproceedings{rakotosaona2020pointcleannet,
  title={POINTCLEANNET: Learning to denoise and remove outliers from dense point clouds},
  author={Rakotosaona, Marie-Julie and La Barbera, Vittorio and Guerrero, Paul and Mitra, Niloy J and Ovsjanikov, Maks},
  booktitle={Computer Graphics Forum},
  volume={39},
  number={1},
  pages={185--203},
  year={2020},
  organization={Wiley Online Library}
}
```

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
