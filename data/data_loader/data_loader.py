import numpy as np
import open3d as o3d
# import laspy
# import pcl

def lidar_loader(lidar_data_path, lidar_dataset_path = None, type_name = None):
    """
        Use different lidar load methods to load lidar data.
        Args:
            lidar_data_path : str
            lidar_dataset_path : str
            type_name: str
        return:
            lidar_data : dict
    """
    if not lidar_dataset_path:
        
        if type_name == 'pcd.bin':
            points = np.fromfile(lidar_data_path, dtype=np.float32, count=-1).reshape([-1, 4])

        if type_name == '.xyz' or type_name == '.xyzn' or type_name == '.xyzrgb' or type_name == '.pts' or type_name == '.ply' or type_name == '.pcd':
            #   .xyz .xyzn .xyzrgb .pts .ply .pcd
            pcd = o3d.io.read_point_cloud(lidar_data_path)
            points = np.asarray(pcd.points)

        # if type_name == 'laspy':
            # data = laspy.read(lidar_data_path)
            # points = np.vstack((data.X, data.Y, data.Z)).transpose()

        # if method_name == 'pcl':
            # points = pcl.load(lidar_data_path)
    
    lidar_data = dict()
    lidar_data["points"] = points
    return points
