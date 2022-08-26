from data.data_filter.data_filter import lidar_filter
import numpy as np

if __name__ == "__main__":

    lidar_data = {"points": np.random.randn(100, 4)}
    lidar_data = lidar_filter( lidar_data, 
                               method = "bilateral_filter", 
                               params = {"radius" : 0.2,
                                         "sigma_d":0.2,
                                         "sigma_n":0.2,
                                         }
                            )
    print(lidar_data["points"].shape)