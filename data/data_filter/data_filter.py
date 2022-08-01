import numpy as np
import open3d as o3d

def remove_statistical_outlier(lidar_data, nb_neighbors, std_ratio):

    points = lidar_data['points']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    filterer_points = np.asarray(filtered_pcd.points)
    if (points.shape[1] > 3):
        filterer_points = np.concatenate( (filterer_points, points[ind, 3:]), axis = -1)

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = filterer_points

    return filtered_lidar_data

def remove_radius_outlier(lidar_data, nb_points, radius):

    points = lidar_data['points']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    filtered_pcd, ind = pcd.remove_radius_outlier(nb_points, radius)

    filterer_points = np.asarray(filtered_pcd.points)
    if (points.shape[1] > 3):
        filterer_points = np.concatenate( (filterer_points, points[ind, 3:]), axis = -1)

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = filterer_points

    return filtered_lidar_data

def voxel_down_sample(lidar_data, voxel_size):

    points = lidar_data['points']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    filtered_pcd = pcd.voxel_down_sample(voxel_size)

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = np.asarray(filtered_pcd.points)
    return filtered_lidar_data

def lidar_filter(lidar_data, method, params):

    if (method == "remove_statistical_outlier"):
        return remove_statistical_outlier(lidar_data, **params)
    elif (method == "remove_radius_outlier"):
        return remove_radius_outlier(lidar_data, **params)
    elif (method == "voxel_down_sample"):
        return voxel_down_sample(lidar_data, **params)
    else:
        raise Exception("Error: `%s` Not implented." % method)

if __name__ == "__main__":

    lidar_data = {"points": np.random.randn(100, 4)}
    lidar_data = lidar_filter( lidar_data, 
                               method = "remove_statistical_outlier", 
                               params = {"nb_neighbors" : 20, "std_ratio" : 0.05}
                            )
    print(lidar_data["points"].shape)

    lidar_data = {"points": np.random.randn(100, 4)}
    lidar_data = lidar_filter( lidar_data, 
                               method = "remove_radius_outlier", 
                               params = {"nb_points" : 10, "radius" : 1}
                            )
    print(lidar_data["points"].shape)

    lidar_data = {"points": np.random.randn(100, 3)}
    lidar_data = lidar_filter( lidar_data, 
                               method = "voxel_down_sample", 
                               params = {"voxel_size" : 1}
                            )
    print(lidar_data["points"].shape)