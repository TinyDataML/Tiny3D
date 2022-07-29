import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pytorch_lightning as pl

from argparse import ArgumentParser

from .net import *
from .loss import *
from deephub.denoisy_model.dmr.utils.dataset import *
from deephub.denoisy_model.dmr.utils.transform import *

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

class PointCloudDenoising(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        try:
            random_mesh = True if hparams.random_mesh else False
        except AttributeError:
            random_mesh = False

        try:
            random_pool = True if hparams.random_pool else False
        except AttributeError:
            random_pool = False

        try:
            random_pool = True if hparams.no_prefilter else False
        except AttributeError:
            no_prefilter = False

        self.model = str_to_class(hparams.net)(
            loss_rec=hparams.loss_rec,
            loss_ds=hparams.loss_ds if hparams.loss_ds != 'None' else None,
            activation=hparams.activation,
            conv_knns=[int(k) for k in hparams.knn.split(',')],
            gpool_use_mlp=True if hparams.gpool_mlp else False,
            dynamic_graph=False if hparams.static_graph else True,
            use_random_mesh=random_mesh,
            use_random_pool=random_pool
        )

        # For validation
        self.cd_loss = ChamferLoss()
        self.emd_loss = EMDLoss(eps=0.005, iters=50)

    def forward(self, pos):
        return self.model(pos)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.hparams.sched_patience, factor=self.hparams.sched_factor, min_lr=self.hparams.min_lr)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):

        # noisifier
        noise_l = self.hparams.noise
        noise_h = self.hparams.noise_high
        if noise_h > noise_l:
            noisifier = AddRandomNoise(std_range=[noise_l, noise_h])
            print('[INFO] Using random noise level.')
        else:
            noisifier = AddNoise(std=self.hparams.noise)

        # Scaling augmentation
        if self.hparams.aug_scale:
            print('[INFO] Scaling augmentation ENABLED.')
            scaler = RandomScale([0.8, 1.2], attr=['pos', 'clean']) # anisotropic scaling doesn't change the direction of normal vectors.
        else:
            print('[INFO] Scaling augmentation DISABLED.')
            scaler = IdentityTransform()

        t = transforms.Compose([
            noisifier,
            RandomRotate(30, attr=['pos', 'clean', 'normal']),  # rotate normal vectors as well.
            scaler,
        ])
        if self.hparams.dataset.find(';') >= 0:
            print('[INFO] Using multiple datasets for training.')
            dataset = MultipleH5Dataset(self.hparams.dataset.split(';'), 'train', normal_name='train_normal', batch_size=self.hparams.batch_size, transform=t, random_get=True, subset_size=self.hparams.subset_size)
        else:
            dataset = H5Dataset(self.hparams.dataset, 'train', normal_name='train_normal', batch_size=self.hparams.batch_size, transform=t)
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        noisifier = AddNoiseForEval([0.01, 0.03, 0.08])
        t = transforms.Compose([
            noisifier,
        ])
        self.val_noisy_item_keys = noisifier.keys
        if self.hparams.dataset.find(';') >= 0:
            dataset_path = self.hparams.dataset.split(';')[0]
            print('[INFO] Validation dataset %s' % dataset_path)
        else:
            dataset_path = self.hparams.dataset
        return DataLoader(
            H5Dataset(dataset_path, 'val', normal_name='val_normal', batch_size=self.hparams.batch_size, transform=t), 
            batch_size=self.hparams.batch_size, shuffle=False
        )


    def training_step(self, batch, batch_idx):
        denoised = self.forward(batch['pos'])
        noiseless = batch['clean']
        normals = batch['normal']

        loss = self.model.get_loss(gts=noiseless, preds=denoised, normals=normals, inputs=batch['pos'])

        output = {
            'loss': loss,
            'log': {
                'loss_train/loss': loss.sum(),
            }
        }
        return output

    def validation_step(self, batch, batch_idx):
        output = {}

        for key in self.val_noisy_item_keys:
            denoised = self.forward(batch[key])
            noiseless = batch['clean']
            output[key + '_emd_loss'] = self.emd_loss(preds=denoised, gts=noiseless).reshape(1)

        return output

    def validation_end(self, outputs):
        self.model.epoch += 1

        output = {
            'val_loss': 0, 
            'log': {'lr': self.optimizer.param_groups[0]['lr']}
        }

        for key in self.val_noisy_item_keys:
            n = len(outputs) * self.hparams.batch_size
            avg_emd_loss = torch.stack([x[key + '_emd_loss'] for x in outputs]).sum() / n
            output['val_loss'] = output['val_loss'] + avg_emd_loss.sum()
            output['log']['loss_val/' + key + '_emd_loss'] = avg_emd_loss.sum()

        return output
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # Network
        parser.add_argument('--net', type=str, default='DenoiseNet')
        parser.add_argument('--gpool_mlp', action='store_true', help='Use MLP instead of single linear layer in the GPool module.')
        parser.add_argument('--knn', type=str, default='8,16,24')
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--static_graph', action='store_true', help='Use static graph convolution instead of dynamic graph (DGCNN).')
        parser.add_argument('--random_mesh', action='store_true', help='Use random mesh instead of regular mesh in the folding layer.')
        parser.add_argument('--random_pool', action='store_true', help='Use random pooling layer instead of differentiable pooling layer.')
        parser.add_argument('--no_prefilter', action='store_true', help='Disable prefiltering.')

        # Dataset and pre-processing
        parser.add_argument('--noise', default=0.02, type=float)
        parser.add_argument('--noise_high', default=0.06, type=float, help='-1 for fixed noise level.')
        parser.add_argument('--aug_scale', action='store_true', help='Enable scaling augmentation.')
        parser.add_argument('--dataset', default='./data/patches_20k_1024.h5;./data/patches_10k_1024.h5;./data/patches_30k_1024.h5;./data/patches_50k_1024.h5;./data/patches_80k_1024.h5', type=str)
        parser.add_argument('--subset_size', default=7000, type=int, help='-1 for unlimited.')
        
        # Batch size
        parser.add_argument('--batch_size', default=8, type=int)

        # Loss
        parser.add_argument('--loss_rec', default='emd', type=str, help='Reconstruction loss.')
        parser.add_argument('--loss_ds', default='cd', type=str, help='Downsample adjustment loss.')

        # Optimizer and scheduler
        parser.add_argument('--learning_rate', default=0.0005, type=float)
        parser.add_argument('--sched_patience', default=10, type=int)
        parser.add_argument('--sched_factor', default=0.5, type=float)
        parser.add_argument('--min_lr', default=1e-5, type=float)        

        return parser