# pretrain_model
# pretrain_model=None
pretrain_model='checkpoint/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

# Trainer params
accelerator='gpu'
devices=1
max_epochs=160
check_val_every_n_epoch=5

# dataset params
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

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
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='ObjectSample',
                db_sampler=dict(
                    data_root='kitti/',
                    info_path='kitti/kitti_dbinfos_train.pkl',
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1],
                        filter_by_min_points=dict(Car=5)),
                    sample_groups=dict(Car=15),
                    classes=['Car'])),
            dict(
                type='ObjectNoise',
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                global_rot_range=[0.0, 0.0],
                rot_range=[-0.15707963267, 0.15707963267]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='DataAugmentor',
                method='random_local_translation_along_y',
                params=[-0.2, 0.2]),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05]),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
            dict(type='PointShuffle'),
            dict(type='DefaultFormatBundle3D', class_names=['Car']),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type='KittiDataset',
        data_root='kitti/',
        ann_file='kitti/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='KittiDataset',
        data_root='kitti/',
        ann_file='kitti/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=2,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car'],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])

