data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='KittiDataset',
        data_root='kitti/',
        ann_file='kitti/kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='DataFilter',
                method='remove_statistical_outlier',
                params={"nb_neighbors": 20, "std_ratio": 0.05}),
            dict(
                type='DataFilter',
                method='remove_radius_outlier',
                params = {"nb_points" : 10, "radius" : 1}),
            dict(
                type='DataAugmentor',
                method='random_flip_along_x'),
            dict(
                type='DataAugmentor',
                method='random_flip_along_y'),
            dict(
                type='DataAugmentor',
                method='global_scaling',
                params=[0.95, 1.05]),
            dict(
                type='DataAugmentor',
                method='random_translation_along_x',
                params=0.2),
            dict(
                type='DataAugmentor',
                method='random_translation_along_y',
                params=0.2),
            dict(
                type='DataAugmentor',
                method='random_translation_along_z',
                params=0.2),
            dict(
                type='DataAugmentor',
                method='global_frustum_dropout_top',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='global_frustum_dropout_bottle',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='global_frustum_dropout_left',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='global_frustum_dropout_right',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='local_scaling',
                params=[0.95, 1.05]),
            dict(
                type='DataAugmentor',
                method='random_local_translation_along_x',
                params=[-0.2, 0.2]),
            dict(
                type='DataAugmentor',
                method='random_local_translation_along_y',
                params=[-0.2, 0.2]),
            dict(
                type='DataAugmentor',
                method='random_local_translation_along_z',
                params=[-0.2, 0.2]),
            dict(
                type='DataAugmentor',
                method='local_frustum_dropout_top',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='local_frustum_dropout_bottle',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='local_frustum_dropout_left',
                params=[0, 0.2]),
            dict(
                type='DataAugmentor',
                method='local_frustum_dropout_right',
                params=[0, 0.2]),
        ]))

