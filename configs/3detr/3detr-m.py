_base_ = [
    '../_base_/models/3detr.py', '../_base_/datasets/vod-lidar-3d-3class.py',
    '../_base_/default_runtime.py'
]



# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/vod/lidar/'
class_names = ['Pedestrian','Cyclist','Car']
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)



# TODO: COPIED FROM 3DSSD
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-1.0471975511965976, 1.0471975511965976]),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.9, 1.1]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=16384),
    # dict(type='Get3detrLabels'),
    dict(type='GetPointcloudMinMax'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','point_cloud_dims_min','point_cloud_dims_max'])

]

# In practice PointPillars also uses a different schedule
# optimizer
lr = 5e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[45, 60])

runner = dict(type='EpochBasedRunner', max_epochs=40)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=40)


#    parser.add_argument("--base_lr", default=5e-4, type=float)
    # parser.add_argument("--warm_lr", default=1e-6, type=float)
    # parser.add_argument("--warm_lr_epochs", default=9, type=int)
    # parser.add_argument("--final_lr", default=1e-6, type=float)
    # parser.add_argument("--lr_scheduler", default="cosine", type=str)
    # parser.add_argument("--weight_decay", default=0.1, type=float)
    # parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    # parser.add_argument(
    #     "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    # )