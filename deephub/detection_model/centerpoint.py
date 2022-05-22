# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from .voxel_encoders import PillarFeatureNet
from .middle_encoders import PointPillarsScatter
from .backbones import SECOND
from .necks import SECONDFPN
from .heads import CenterHead
import torch

class Centerpoint(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 init_cfg=None):
        super(Centerpoint, self).__init__(init_cfg=init_cfg)
        self.pts_voxel_encoder = PillarFeatureNet(
            in_channels=4,
            feat_channels=[64],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        )
        self.pts_middle_encoder = PointPillarsScatter(
            in_channels=64,
            output_shape=[512,512]
        )
        self.pts_backbone = SECOND(
            in_channels=64,
            out_channels=[64, 128, 256]
        )
        self.pts_neck = SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
            use_conv_for_no_stride=True
        )
        self.pts_bbox_head = CenterHead(
            in_channels=384,
            tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])],
            common_heads=dict(
                   reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
        )

    def forward(self,
                voxels,
                num_points,
                coors):
        """Test function without augmentaiton. Rewrite this func to remove model
            post process.

            Args:
                voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
                num_points (torch.Tensor): Number of points in each pillar.
                coors (torch.Tensor): Coordinates of each voxel.

            Returns:
                List: Result of model.
            """
        x = self.extract_feat(voxels, num_points, coors)
        outs = self.pts_bbox_head(x)
        bbox_preds, scores, dir_scores = [], [], []
        for task_res in outs:
            bbox_preds.append(task_res[0]['reg'])
            bbox_preds.append(task_res[0]['height'])
            bbox_preds.append(task_res[0]['dim'])
            if 'vel' in task_res[0].keys():
                bbox_preds.append(task_res[0]['vel'])
            scores.append(task_res[0]['heatmap'])
            dir_scores.append(task_res[0]['rot'])
        bbox_preds = torch.cat(bbox_preds, dim=1)
        scores = torch.cat(scores, dim=1)
        dir_scores = torch.cat(dir_scores, dim=1)
        return scores, bbox_preds, dir_scores

    def extract_feat(self,
                     voxels,
                     num_points,
                     coors):
        """Extract features from points. Rewrite this func to remove voxelize op.

        Args:
            voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features from points.
        """
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1  # refactor
        assert batch_size == 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

