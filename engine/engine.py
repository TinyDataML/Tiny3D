import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

def fit(train_dataset, val_dataset, model, batchsize=100, epoch=1, devices=4, accelerator="gpu", strategy="deepspeed_stage_2", precision=32):
    """
        Runs the full optimization routine.
        Args:
            trainval_dataset:  dataset to be fit
            model:   model to be fit
            batchsize:  batchsize
            epoch:  epoch
            devices:   gpu/cpu number
            accelerator: training devices.
            strategy: training type
            precision: training precision.
        return:
            demoised lidar_data
        Reference:
            https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/
    """

    trainer = pl.Trainer(limit_train_batches=batchsize, max_epochs=epoch)
    trainer.fit(model, DataLoader(train_dataset), DataLoader(val_dataset))

    return model

def test(test_dataset, model):
    """
        Runs the full test routine.
        Args:
            test_dataseta: dataset to be test
            model: model to be test
        return:
            model
        Reference:
            https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/
    """
    
    trainer = pl.Trainer()
    trainer.test(model, dataloaders=DataLoader(test_dataset))

    return model
