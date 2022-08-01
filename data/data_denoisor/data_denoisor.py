import numpy as np
import torch

from .denoisor_dmr_utils import run_denoise_large_pointcloud, run_denoise_middle_pointcloud, run_denoise
from .denoisor_pcpnet_utils import ResPCPNet

def lidar_denoisor(lidar_data, method):
    """
        Use different lidar denoise methods to denoise lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            demoised lidar_data
        Reference:
            https://github.com/luost26/DMRDenoise
            https://github.com/mrakotosaon/pointcleannet
    """
    points = lidar_data['points']

    if method == 'pcp':
        param_filename = 'deephub/denoisy_model/pcp/pretrained/denoisingModel/PointCleanNet_params.pth'
        model_filename = 'deephub/denoisy_model/pcp/pretrained/denoisingModel/PointCleanNet_model.pth'
        trainopt = torch.load(param_filename)
        pred_dim = 0
        output_pred_ind = []
        for o in trainopt.outputs:
            if o in ['clean_points']:
                output_pred_ind.append(pred_dim)
                pred_dim += 3
            else:
                raise ValueError('Unknown output: %s' % (o))

        regressor = ResPCPNet(
            num_points=trainopt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=trainopt.use_point_stn,
            use_feat_stn=trainopt.use_feat_stn,
            sym_op=trainopt.sym_op,
            point_tuple=trainopt.point_tuple)
        regressor.load_state_dict(torch.load(model_filename))
        
        pred, trans, _, _ = regressor(points)
        patch_radiuses=torch.FloatTensor([0.05])
       
        denoised = pred
      
    if method == 'dmr':
        num_points = points.shape[0]
        if num_points >= 120000:
            print('[INFO] Denoising large point cloud.')
            denoised, downsampled = run_denoise_large_pointcloud(
                pc=points,
                cluster_size=30000,
                patch_size=1000,
                ckpt='deephub/denoisy_model/dmr/pretrained/supervised/epoch=153.ckpt',
                device='cuda:0',
                random_state=0,
                expand_knn=16
            )
        elif num_points >= 60000:
            print('[INFO] Denoising middle-sized point cloud.')
            denoised, downsampled = run_denoise_middle_pointcloud(
                pc=points,
                num_splits=2,
                patch_size=1000,
                ckpt='deephub/denoisy_model/dmr/pretrained/supervised/epoch=153.ckpt',
                device='cuda:0',
                random_state=0,
                expand_knn=16
            )
        elif num_points >= 10000:
            print('[INFO] Denoising regular-sized point cloud.')
            denoised, downsampled = run_denoise(
                pc=points,
                patch_size=1000,
                ckpt='deephub/denoisy_model/dmr/pretrained/supervised/epoch=153.ckpt',
                device='cuda:0',
                random_state=0,
                expand_knn=16
            )
        else:
            assert False, "Our pretrained model does not support point clouds with less than 10K points."

    lidar_data['points'] = denoised

    return lidar_data
