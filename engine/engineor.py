import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from engine.pointpillars_engine import Pointpillars_engine
import pytorch_lightning.callbacks as plc
from functools import partial
from mmcv.parallel import collate


def fit(dataset_train, dataset_val, torch_model, epoch=80, devices=1, accelerator="gpu", check_val_every_n_epoch=5, pretrain=None):
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

    model = Pointpillars_engine(torch_model)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(collate, samples_per_gpu=1))
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate, samples_per_gpu=1))

    # load callbacks
    callbacks = load_callbacks()

    # load pretrain state dict
    if pretrain != None:
        checkpoint = torch.load(pretrain)
        model.torch_model.load_state_dict(checkpoint["state_dict"])
        print('loading pretrain from:  ' + pretrain)

    # init Trainer
    # trainer = Trainer.from_argparse_args(args)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=epoch, deterministic=True,
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      num_sanity_val_steps=0, callbacks=callbacks)

    trainer.fit(model=model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

def eval(dataset_val, torch_model, weights=None, accelerator='gpu', devices=1):
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

    model = Pointpillars_engine(torch_model)
    # load weights
    if weights != None:
        checkpoint = torch.load(weights)
        model.torch_model.load_state_dict(checkpoint["state_dict"])
        print('loading pretrain from:  ' + weights)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate, samples_per_gpu=1))

    # init Trainer
    # trainer = Trainer.from_argparse_args(args)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, deterministic=True)
    trainer.validate(model=model, dataloaders=data_loader_val)

def predict(dataset_val, torch_model, weights=None, accelerator='gpu', devices=1):
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

    model = Pointpillars_engine(torch_model)
    # load weights
    if weights != None:
        checkpoint = torch.load(weights)
        model.torch_model.load_state_dict(checkpoint["state_dict"])
        print('loading pretrain from:  ' + weights)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate, samples_per_gpu=1))

    # init Trainer
    # trainer = Trainer.from_argparse_args(args)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, deterministic=True)
    predicts = trainer.predict(model=model, dataloaders=data_loader_val, return_predictions=True)
    return predicts

def inference(pcd_data, torch_model, weights=None, accelerator='gpu', devices=1):
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

    model = Pointpillars_engine(torch_model)
    # load weights
    if weights != None:
        checkpoint = torch.load(weights)
        model.torch_model.load_state_dict(checkpoint["state_dict"])
        print('loading pretrain from:  ' + weights)

    model.cuda()
    model.eval()

    predict = model(pcd_data['img_metas'][0], pcd_data['points'][0])
    return predict



def load_callbacks():
    callbacks = []

    callbacks.append(plc.ModelCheckpoint(
        monitor='ap',
        filename='best-{epoch:02d}-{ap:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks