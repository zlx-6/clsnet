dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
log_level = 'INFO'
work_dir = './work_dirs\cifar10'
gpu_ids = range(0, 1)
