import sys, os
sys.path.append(".")
import unittest

import point_cloud_utils as pcu
from data.data_filter.data_filter import lidar_filter
import numpy as np
from matplotlib import pyplot as plt

def chamfer_distance(points1, points2):

    return pcu.chamfer_distance(points1, points2)

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
    #plt.show()
    plt.pause(2)

class TestDownSample:

    def setUp(self) -> None:

        plt.ion()
        self.points = np.load("./test/test_data_ops/airplane.npy")
        
        print("Original number of points:",  self.points.shape[0])
        draw_pointcloud(self.points, title = "original point cloud")
    
    def tearDown(self) -> None:

        plt.ioff()
        return super().tearDown()

    def test_voxel_down_sample(self):
        '''
            Test voxel downsampling method.
        '''
        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "voxel_down_sample", 
                                            params = {"voxel_size" : 0.2,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "voxel downsample")
        print("Number of points (voxel downsample): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] < lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

    def test_random_sample(self):
        '''
            Test random downsampling method.
        '''

        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "random_sample", 
                                            params = {"n_points" : 256,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "random downsample")
        print("Number of points (random downsample): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] < lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

    def test_uniform_down_sample(self):
        '''
            Test uniform downsampling method.
        '''

        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "uniform_down_sample", 
                                            params = {"voxel_size" : 0.2,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "uniform downsample")
        print("Number of points (uniform downsample): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] < lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

    def test_farthest_point_sample(self):
        '''
            Test farthest point downsampling method.
        '''

        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "farthest_point_sample", 
                                            params = {"n_points" : 512,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "farthest point sample")
        print("Number of points (farthest point sample): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] < lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

class TestOutlierRemoval():

    def setUp(self) -> None:

        plt.ion()
        self.points = np.load("./test/test_data_ops/airplane.npy")
        num_points, num_channels = self.points.shape

        # add outliers
        self.points = np.concatenate(
            (self.points, 
             np.random.randn(20, num_channels) * self.points.std(axis = 0, keepdims = True)
            ),
            axis = 0,
        )
        
        print("Original number of points:",  num_points + 20)
        draw_pointcloud(self.points, title = "original point cloud")
    
    def tearDown(self) -> None:

        plt.ioff()
        return super().tearDown()

    def test_remove_radius_outlier(self):
        '''
            Test radius outlier removal method.
        '''
        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "remove_radius_outlier", 
                                            params = {"nb_points" : 10,
                                                      "radius" : 0.3,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "remove radius outlier")
        print("Number of points (remove radius outlier): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] <= lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

    def test_remove_statistical_outlier(self):
        '''
            Test statistical outlier removal method.
        '''
        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "remove_statistical_outlier", 
                                            params = {"nb_neighbors" : 20,
                                                      "std_ratio" : 2.0,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "remove statistical outlier")
        print("Number of points (remove statistical outlier): ", filtered_lidar_data["points"].shape[0])
        assert filtered_lidar_data["points"].shape[0] <= lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

class TestSmooth(unittest.TestCase):

    def setUp(self) -> None:

        plt.ion()
        self.original_points = np.load("./test/test_data_ops/airplane.npy")
        
        # add noise
        self.points = self.original_points.copy()
        self.points += 0.05 * np.random.randn(*self.points.shape) * np.std(self.points, axis = 0, keepdims = True)
        
        print("CD (noisy):",  chamfer_distance(self.points, self.original_points))
        draw_pointcloud(self.points, title = "original point cloud")
    
    def tearDown(self) -> None:

        plt.ioff()
        return super().tearDown()

    # def test_bilateral_filter(self):
    #     '''
    #         Test bilateral filtering method.
    #     '''
    #     lidar_data = {"points": self.points}
    #     filtered_lidar_data = lidar_filter( lidar_data, 
    #                                         method = "bilateral_filter", 
    #                                         params = {"radius" : 0.5,
    #                                                   "sigma_d" : 1.0,
    #                                                   "sigma_n" : 1.0,
    #                                                   "knn" : 20,
    #                                                 }
    #                                     )
    #     draw_pointcloud(filtered_lidar_data["points"], title = "bilateral filter")
    #     print("CD (bilateral filter):",  chamfer_distance(filtered_lidar_data["points"], self.original_points))
    #     assert filtered_lidar_data["points"].shape[0] <= lidar_data["points"].shape[0]
    #     assert filtered_lidar_data["points"].shape[1] >= 3

    def test_weighted_local_optimal_projection(self):
        '''
            Test weighted local optimal projection method.
        '''
        lidar_data = {"points": self.points}
        filtered_lidar_data = lidar_filter( lidar_data, 
                                            method = "weighted_local_optimal_projection", 
                                            params = {"n_points" : 1024,
                                                      "num_iter" : 100,
                                                    }
                                        )
        draw_pointcloud(filtered_lidar_data["points"], title = "wlop")
        print("CD (wlop):",  chamfer_distance(filtered_lidar_data["points"], self.original_points))
        assert filtered_lidar_data["points"].shape[0] <= lidar_data["points"].shape[0]
        assert filtered_lidar_data["points"].shape[1] >= 3

if __name__ == "__main__":
        
    unittest.main()