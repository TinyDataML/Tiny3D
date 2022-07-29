from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from mmcv import Config
from mmdet3d.datasets import build_dataset

from model.model_deployor.deployor_utils import create_input
from engine.pointpillars_engine import Pointpillars_engine
from deephub.detection_model import Pointpillars
from engine import fit, eval, predict, inference

def main(args):
    pl.seed_everything(args.seed, workers=True)

    if args.model_name == "pointpillars":
        torch_model = Pointpillars()

    cfg = Config.fromfile(args.config)
    dataset_train = build_dataset(cfg.data.train)
    dataset_val = build_dataset(cfg.data.val)

    if args.mode == 'fit':
        fit(dataset_train=dataset_train, dataset_val=dataset_val, torch_model=torch_model, epoch=cfg.max_epochs,
            devices=cfg.devices, accelerator=cfg.accelerator, check_val_every_n_epoch=cfg.check_val_every_n_epoch,
            pretrain=cfg.pretrain_model)
    elif args.mode == 'eval':
        eval(dataset_val=dataset_val, torch_model=torch_model, weights=cfg.pretrain_model, accelerator=cfg.accelerator,
             devices=cfg.devices)
    elif args.mode == 'predict':
        predicts = predict(dataset_val=dataset_val, torch_model=torch_model, weights=cfg.pretrain_model, accelerator=cfg.accelerator,
             devices=cfg.devices)
    elif args.mode == 'inference':
        data, model_inputs = create_input('test/data_tobe_tested/kitti/kitti_000008.bin', 'kitti', 'pointpillars',
                                          'cuda:0')
        pcd_result = inference(pcd_data=data, torch_model=torch_model, weights=cfg.pretrain_model,
                            accelerator=cfg.accelerator,
                            devices=cfg.devices)
    elif args.mode == 'infer_production':
        checkpoint = torch.load("lightning_logs/version_9/checkpoints/epoch=3-step=14848.ckpt")
        state_dict = checkpoint["state_dict"]

        # update keys by dropping `torch_model.`
        for key in list(state_dict):
            state_dict[key.replace("torch_model.", "")] = state_dict.pop(key)

        torch_model.load_state_dict(state_dict)
        torch_model.cuda()
        torch_model.eval()
        data, model_inputs = create_input('test/data_tobe_tested/kitti/kitti_000008.bin', 'kitti', 'pointpillars',
                                          'cuda:0')
        torch_out = torch_model(model_inputs[0], model_inputs[1], model_inputs[2])

    print('a')




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default="pointpillars", help="choice running model")
    parser.add_argument("--mode", type=str, default="fit", choices=['fit', 'eval', 'predict',
                                                                      'inference', 'infer_production'], help="choice running mode")
    parser.add_argument("--config", type=str, help="train or inference")

    # Get model name
    temp_args, _ = parser.parse_known_args()

    # get model params
    if temp_args.model_name == "pointpillars":
        parser = Pointpillars_engine.add_model_specific_args(parser)

    # # Add pytorch lightning's args to parser as a group.
    # parser = Trainer.add_argparse_args(parser)

    # Set seed and deterministic to ensure full reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#reproducibility
    parser.set_defaults(seed=131)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    main(args)
