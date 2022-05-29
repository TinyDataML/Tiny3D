import numpy as np
import open3d as o3d
import laspy
import pcl

def lidar_loader(lidar_data_path,lidar_dataset_path=None, method_name=None):
    """
        Use different lidar load methods to load lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            points
        Reference:
            https://github.com/luost26/DMRDenoise
    """
    if lidar_dataset_path =='None':
        if method_name == 'numpy':
            points = np.fromfile(lidar_data_path, dtype=np.float32, count=-1).reshape([-1, 4])

        if method_name == 'open3d':
            # 文件类型是自动识别的，支持 .xyz .xyzn .xyzrgb .pts .ply .pcd类型的文件
            pcd = o3d.io.read_point_cloud(lidar_data_path)

        if method_name == 'laspy':
            data = laspy.read(lidar_data_path)
            pcloud = np.vstack((data.X, data.Y, data.Z)).transpose()

        if method_name == 'pcl':
            pt = pcl.load(lidar_data_path)
    return points

def read_pcd_file(pcd, test_pipeline, device, box_type_3d):
    """Read data from pcd file and run test pipeline.

    Args:
        pcd (str): Pcd file path.
        device (str): A string specifying device type.

    Returns:
        dict: meta information for the input pcd.
    """