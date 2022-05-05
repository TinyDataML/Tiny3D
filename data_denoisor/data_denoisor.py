import numpy as np

from denoisor_dmr_utils import run_denoise_large_pointcloud, run_denoise_middle_pointcloud, run_denoise

def lidar_denoiser(lidar_data, method):
    """
        Use different lidar denoise methods to denoise lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            demoised lidar_data
        Reference:
            https://github.com/luost26/DMRDenoise
    """
    points = lidar_data['points']

    if method == 'dmr':
        num_points = points.shape[0]
        if num_points >= 120000:
            print('[INFO] Denoising large point cloud.')
            denoised, downsampled = run_denoise_large_pointcloud(
                pc=points,
                cluster_size=30000,
                patch_size=1000,
                ckpt='./dmr/pretrained/supervised/epoch=153.ckpt',
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
                ckpt='./dmr/pretrained/supervised/epoch=153.ckpt',
                device='cuda:0',
                random_state=0,
                expand_knn=16
            )
        elif num_points >= 10000:
            print('[INFO] Denoising regular-sized point cloud.')
            denoised, downsampled = run_denoise(
                pc=points,
                patch_size=1000,
                ckpt='./dmr/pretrained/supervised/epoch=153.ckpt',
                device='cuda:0',
                random_state=0,
                expand_knn=16
            )

    lidar_data['points'] = denoised

    return lidar_data
