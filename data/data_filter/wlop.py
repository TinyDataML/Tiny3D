import numpy as np

def compute_theta(r, h):

    return np.exp(-r ** 2 / h**2 * 16)

def weighted_local_optimal_projector(
    lidar_data, 
    n_points = 1024, 
    num_iter = 25, 
    h = None
):

    '''
        Reference:
            https://github.com/jinseokbae/WLOP-based-PointCloudDenoiser
            Huang et al.,2009, Consolidation of Unorganized Point Clouds for Surface Reconstruction
    '''

    points = lidar_data['points'][:, :3]
    n, m = n_points, points.shape[0]

    d_bb = np.sqrt( 
        ((points.max(axis = 0) - points.min(axis = 0)) ** 2).sum()
    )
    if h is None:
        h = 4*np.sqrt(d_bb)
    X = np.random.rand(n_points, 3) * \
            ( points.max(axis = 0, keepdims = True) - points.min(axis = 0, keepdims = True)) \
            + points.min(axis = 0, keepdims = True)
    
    pairwise_dis_pp = np.linalg.norm(points[:, :, None] - points.T[None, :, :], axis = 1)
    v = compute_theta(pairwise_dis_pp, h).sum(axis = 1) # (m, )

    for _ in range(num_iter):

        newX = np.empty_like(X)

        pairwise_displacement_xx = np.transpose(X[:, :, None] - X.T[None, :, :], (0, 2, 1)) # (n, n, 3)
        pairwise_dis_xx = np.linalg.norm(pairwise_displacement_xx, axis = 2) # (n, n)
        w = compute_theta(pairwise_dis_xx, h).sum(axis = 1) # (n, )

        pairwise_dis_xp = np.linalg.norm(X[:, :, None] - points.T[None, :, :], axis = 1) # (n, m)
        alpha = compute_theta(pairwise_dis_xp, h) 
        alpha = alpha / (pairwise_dis_xp * v[None, :]) # (n, m)
        alpha = alpha / alpha.sum(axis = 1, keepdims = True) 

        beta = - compute_theta(pairwise_dis_xx, h) # (n, n)
        beta = beta / pairwise_dis_xx * w[None, :] 
        np.fill_diagonal(beta, 0) # (n, n)
        beta = beta / beta.sum(axis = 1, keepdims = True) 

        newX = (alpha @ points) \
               + (beta[:, :, None] * pairwise_displacement_xx).sum(axis = 1)
        X = newX

    filtered_lidar_data = lidar_data
    filtered_lidar_data["points"] = X
    return filtered_lidar_data