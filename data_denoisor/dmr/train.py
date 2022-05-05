import os
import torch

import numpy as np
import random

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

from models.denoise import PointCloudDenoising

def main(hparams):
    
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    module = PointCloudDenoising(hparams)

    if hparams.debug:
        trainer = Trainer(
            gpus=hparams.n_gpu,
            fast_dev_run=True,
            logger=False,
            checkpoint_callback=False,
            distributed_backend='dp'
        )
    else:
        trainer = Trainer(
            gpus=hparams.n_gpu,
            early_stop_callback=None,
            distributed_backend='dp',
        )
        os.makedirs('./lightning_logs', exist_ok=True)
        os.makedirs(trainer.logger.log_dir)
        trainer.checkpoint_callback = ModelCheckpoint(
            filepath = trainer.logger.log_dir,
            save_top_k=-1
        )

    trainer.fit(module)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2020)
    parser = PointCloudDenoising.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
