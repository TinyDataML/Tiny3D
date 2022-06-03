# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from .voxel_encoders import PillarFeatureNet
from .middle_encoders import PointPillarsScatter
from .backbones import SECOND
from .necks import SECONDFPN
from .heads import Anchor3DHead

class Pointpillars(BaseModule):
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
        super(Pointpillars, self).__init__(init_cfg=init_cfg)
        self.voxel_encoder = PillarFeatureNet(
            feat_channels=[64],
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
        )
        self.middle_encoder = PointPillarsScatter(
            in_channels=64,
            output_shape=[496,432]
        )
        self.backbone = SECOND(
            in_channels=64,
            out_channels=[64, 128, 256]
        )
        self.neck = SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128]
        )
        self.bbox_head = Anchor3DHead(
            num_classes=1,
            in_channels=384,
            feat_channels=384
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
        bbox_preds, scores, dir_scores = self.bbox_head(x)
        return bbox_preds, scores, dir_scores

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
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1  # refactor
        # assert batch_size == 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

