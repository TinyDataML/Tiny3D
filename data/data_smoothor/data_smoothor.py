import pcl

def lidar_smooth(lidar_data, method):
    """
        Use different data smooth methods to smooth lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            smoothed lidar_data
        Reference:
            https://github.com/strawlab/python-pcl
    """
    points = lidar_data['points']
    coors = points[:,0:3].numpy()

    if method == 'KNN':
        p = pcl.PointCloud(coors)
        fil = p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        coors_filter = fil.filter()
        # coors_index = coors.index(fil.filter().numpy())
        # points = points[coors_index]
        points = coors_filter

    lidar_data['points'] = points

    return lidar_data
