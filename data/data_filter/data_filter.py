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
    
    _, _, ind_list = pcd.voxel_down_sample_and_trace(
        voxel_size = voxel_size, 
        min_bound = pcd.get_min_bound() - voxel_size * 0.5, 
        max_bound = pcd.get_max_bound() + voxel_size * 0.5
    )

    filtered_points = np.stack(
        [points[ind, ...].mean(axis = 0) for ind in ind_list], 
        axis = 0
    )

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = filtered_points
    return filtered_lidar_data

def uniform_down_sample(lidar_data, voxel_size):

    points = lidar_data['points']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    _, _, ind_list = pcd.voxel_down_sample_and_trace(
        voxel_size = voxel_size, 
        min_bound = pcd.get_min_bound() - voxel_size * 0.5, 
        max_bound = pcd.get_max_bound() + voxel_size * 0.5
    )

    survived = [
        ind[np.linalg.norm((points[ind, :3] - points[ind, :3].mean(axis = 0, keepdims = True)), 
                            axis = 1
                          ).argmin()]
        for ind in ind_list
    ]

    filtered_points = points[survived, ...]

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = filtered_points
    return filtered_lidar_data

def farthest_point_sample(lidar_data, n_points):
    
    points = lidar_data['points']

    N = points.shape[0]
    xyz = points[:,:3]

    idx = np.zeros((n_points,), dtype = np.int32)
    distance = np.ones((N,)) * np.inf
    farthest = np.random.randint(0, N)

    for i in range(n_points):
        idx[i] = farthest
        centroid = xyz[farthest, :]
        dist_centroid = np.sum((xyz - centroid) ** 2, axis = -1)
        distance = np.minimum(dist_centroid, distance)
        farthest = np.argmax(distance, -1)

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = points[idx, ...]
    return filtered_lidar_data

def random_sample(lidar_data, n_points):
    
    points = lidar_data['points']

    N = points.shape[0]
    selected = np.random.permutation(N)[:n_points]

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = points[selected, ...]
    return filtered_lidar_data

def bilateral_filter(lidar_data, radius, sigma_d, sigma_n, knn = 30):

    points = lidar_data['points']
    filtered_points = np.copy(points)
    N = points.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(N):

        # find neighbours
        _, idx, _ = pcd_tree.search_hybrid_vector_3d(pcd.points[i], radius = radius, max_nn = 10)
        idx = np.asarray(idx)
        idx = idx[idx != i]
        neighbours = points[idx, :]

        if (neighbours.shape[0] < 3):
            continue

        # normal estimation
        _, _, V = np.linalg.svd(neighbours - neighbours.mean(axis = 0, keepdims = True), 
                                full_matrices = False)
        normal = V[2, :]

        # compute distances
        displacement = neighbours - points[[i], :]
        dis_d = np.linalg.norm(displacement, axis = 1)
        dis_n = np.dot(displacement, normal)
        weight = np.exp(- dis_d ** 2 / (2 * sigma_d**2) ) * \
            np.exp(- dis_n ** 2 / (2 * sigma_n**2) )
        filtered_points[i, :3] += (weight * dis_n).sum() / weight.sum() * normal

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = filtered_points
    return filtered_lidar_data


def lidar_filter(lidar_data, method, params):

    if (method == "remove_statistical_outlier"):
        return remove_statistical_outlier(lidar_data, **params)
    elif (method == "remove_radius_outlier"):
        return remove_radius_outlier(lidar_data, **params)
    elif (method == "voxel_down_sample"):
        return voxel_down_sample(lidar_data, **params)
    elif (method == "uniform_down_sample"):
        return uniform_down_sample(lidar_data, **params)
    elif (method == "farthest_point_sample"):
        return farthest_point_sample(lidar_data, **params)
    elif (method == "random_sample"):
        return random_sample(lidar_data, **params)
    elif (method == "bilateral_filter"):
        return bilateral_filter(lidar_data, **params)
    else:
        raise Exception("Error: `%s` Not implented." % method)

