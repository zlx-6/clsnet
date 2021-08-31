_base_ = [
    '../_base_/models/mobilenet_v3_small_cifar.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

lr_config = dict(policy='step',warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)
