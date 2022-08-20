import sys, os
sys.path.append(".")
import unittest

from data.data_filter.data_filter import lidar_filter
import numpy as np
from matplotlib import pyplot as plt

def draw_pointcloud(points, colors = None, title = None):

    '''
        points: numpy array, of shape (n, 3)
              [(x, y, z), ...]
        colors: numpy array, of shape (n, 3)
                [(r, g, b), ...]
        title: str.
    '''

    extrinsic_mat = np.array(
        [-0.66836757203249464,-0.40034499349702946,0.62690403957643437,0.0,
		-0.023664847020242345,-0.83093287073494904,-0.55586917466944075,0.0,
		0.74345461439881899,-0.38636051883446904,0.54589448230868143,0.0,
		0.06979788144697574,0.30204396056447935,7.5263119146523731,1.0]
    ).reshape(4, 4)    

    intrinsic_mat = np.array(
        [935.30743608719376,       0.0,       0.0,
        0.0,       935.30743608719376,       0.0,
        959.5,       539.5,       1.0]
    ).reshape(3, 3)

    plt.figure()
    ax = plt.axes()
    ax.invert_yaxis()
    ax.set_aspect(1)
    
    n = points.shape[0]
    points = np.concatenate( [points, np.ones((n, 1))], axis = 1 )
    points = (extrinsic_mat @ points.T).T
    points = np.concatenate( [points[:, :2], np.ones((n, 1))], axis = 1 )
    points = (intrinsic_mat @ points.T).T
    plt.scatter(points[:, 0], points[:, 1], c = colors, marker = ".", s = 10)
    if title:
        plt.title(title)
    plt.show()


class TestDownSample(unittest.TestCase):

    def test_voxel_down_sample(self):
        '''
            Test voxel downsampling method.
        '''
        points = np.load("./test/test_data_ops/airplane.npy")
        # add noise
        # points += 0.05 * np.random.randn(*points.shape) * np.abs(points).max(axis = 0, keepdims = True)
        print("Original number of points:", points.shape[0])
        lidar_data = {"points": points}

        draw_pointcloud(lidar_data["points"], title = "before")

        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "voxel_down_sample", 
                                            params = {"voxel_size" : 0.2,
                                                    }
                                        )
        print(filtered_lidar_data["points"].shape)

        draw_pointcloud(filtered_lidar_data["points"], title = "after")

        assert filtered_lidar_data["points"].shape != lidar_data["points"].shape


if __name__ == "__main__":

    unittest.main()