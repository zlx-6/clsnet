from mmcls.datasets.builder import build_dataloader, build_dataset
from mmcls.cvcore.utils.logging import get_logger
from mmcls.cvcore.parallel import MMDataParallel
from mmcls.cvcore.runner import build_optimizer,build_runner
from mmcls.cvcore.runner.evaluation import EvalHook

import random
import warnings
import numpy as np
import torch

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model,dataset,cfg,distributed=False,validate=False,timestamp=None,device='cuda',meta=None):
    logger =get_logger(cfg.log_level)
    
    dataset = dataset if isinstance(dataset,(tuple,list)) else [dataset]
    print(len(cfg.gpu_ids))
    data_loaders = [
        build_dataloader(ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        num_gpus=len(cfg.gpu_ids),
                        dist=distributed,
                        round_up=True,
                        seed=cfg.seed) for ds in dataset ]
    if distributed:
        pass#不支持分布式训练啊现在
    else:
        if device == 'cuda':
            print('use cuda')
            model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        elif device =='cpu':
            model = model.cpu()
        else:
            raise ValueError(f'unsupported device name {device}')
    
    optimizer = build_optimizer(model,cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(cfg.runner,
                        default_args=dict(
                            model=model,
                            batch_processor=None,
                            optimizer=optimizer,
                            work_dir=cfg.work_dir,
                            logger=logger,
                            meta=meta))
    runner.timestamp = timestamp

    #fp16的设置,这一部分先省略
    fp16_cfg = cfg.get('fp16',None)
    if fp16_cfg is not None:
        pass
    elif distributed and 'type' not in cfg.optimizer_config:#省略
        optimizer_config = None
    else:
        optimizer_config = cfg.optimizer_config
    
    #注册一些必要的hook
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config',None),
        custom_hooks_config=cfg.get('custom_hooks',None))

    if distributed:
        print('do not support distributed!, you must set the distributed is False')

    if validate:
        val_dataset = build_dataset(cfg.data.val,dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            round_up=True)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook =  EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders,cfg.workflow)    
