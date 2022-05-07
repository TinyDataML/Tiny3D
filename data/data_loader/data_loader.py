import numpy as np

def lidar_loader(lidar_data_path):
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
    points = np.fromfile(str("lidar_path"), dtype=np.float32, count=-1).reshape([-1, 4])

    return points
