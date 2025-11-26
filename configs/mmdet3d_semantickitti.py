"""
MMDetection3D SemanticKITTI预训练模型配置
用于3D点云几何特征提取
"""
from mmcv import Config

# 基础配置 - 使用PointNet++或SparseUNet作为backbone
# 注意：这是一个简化的配置，实际使用时需要根据MMDetection3D的具体模型结构调整

# 模型配置
model = dict(
    type='EncoderDecoder3D',  # 3D编码器-解码器结构
    backbone=dict(
        type='PointNet2SASSG',  # PointNet++作为backbone
        in_channels=3,  # 输入特征维度（x, y, z）
        num_points=(2048, 1024, 512, 256),  # 采样点数
        radius=(0.2, 0.4, 0.8, 1.2),  # 搜索半径
        num_samples=(64, 32, 16, 16),  # 每个点的采样数
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256), (128, 128, 256)),
        fp_channels=((256, 256), (256, 128), (128, 128, 128)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False
        )
    ),
    decode_head=dict(
        type='PointHead',
        in_channels=128,  # 解码器输入特征维度
        channels=128,  # 解码器中间特征维度
        num_classes=19,  # SemanticKITTI类别数（实际使用时可能不同）
        dropout_ratio=0.1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0
        )
    ),
    # 训练和测试设置
    train_cfg=None,
    test_cfg=dict(mode='whole')  # 使用整个点云进行推理
)

# 数据配置（用于参考，实际使用时可能需要调整）
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='SemanticKITTIDataset',
        data_root='data/semantickitti/',
        ann_file='data/semantickitti/semantickitti_infos_train.pkl',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=3),
            dict(type='PointSegClassMapping'),
            dict(type='DefaultFormatBundle3D', class_names=None),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ],
        classes=None,
        test_mode=False
    )
)

# 优化器配置（用于参考）
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率调度器配置（用于参考）
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False
)

# 运行配置（用于参考）
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# 注意：这个配置文件是一个模板
# 实际使用时需要根据MMDetection3D的具体版本和模型结构调整
# 建议从MMDetection3D官方仓库获取SemanticKITTI的配置文件作为基础


