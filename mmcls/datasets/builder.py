from mmcls.cvcore import Registry,build_from_cfg
from mmcls.cvcore.runner import get_dist_info
from mmcls.cvcore.parallel import collate

from functools import partial
from distutils.version import LooseVersion
from torch.utils.data import DataLoader
import numpy as np
import random
import torch

PIPELINES = Registry('pipeline')
DATASETS = Registry('datasets')

def build_dataset(cfg, default_args=None):
    
    #只支持建立一个简单的不进行任何操作的数据集
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

def build_dataloader(dataset,samples_per_gpu,workers_per_gpu,num_gpus=1,dist=False,shuffle=True,
                        round_up=True,seed=None,pin_memory=True,persistent_workers=True,**kwargs):
        
    rank,world_size = get_dist_info()#0,1
    if dist:   #是否使用分布式训练，这里默认改成了False，即不使用分布式采样，后面会加上
        pass
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

        init_fn = partial(worker_init_fn,num_workers=num_workers,rank=rank,seed=seed) if seed is not None else None

    if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)
    return data_loader

def worker_init_fn(worker_id,num_workers,rank,seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
