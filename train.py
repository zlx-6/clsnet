import copy
import os
import os.path as osp
import argparse
import torch
import time

#mmcv.Config
from mmcls.cvcore.utils import DictAction,Config,mkdir_or_exist
from mmcls.utils import get_root_logger,collect_env
from mmcls.apis import set_random_seed,train_model
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset



def parse_args():
    parser= argparse.ArgumentParser(description="Tran a classification model")
    parser.add_argument('--config',default='config\\resnet\\resnet18_cifar10.py',help ='train config file')
    parser.add_argument('--work-dir',help='the dir to save the log and models')
    parser.add_argument('--resume-from',help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()#创建一个互斥组。 argparse 将会确保互斥组中只有一个参数在命令行中可用
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')#一个字典参数，这里还不太了解
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)#从py文件获得配置参数
    if args.options is not None:
        cfg.merge_from_dict(args.options)#从args的options中再获得一些参数
    if cfg.get('cudnn_benchmark',False):
        torch.backends.cudnn.benchmark = True
    #设置保存文件的路径
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
    else:
        #只支持非分布式训练,后面再添加分布式
        distributed = False

    #创建工作文件夹
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    #保存配置成为一个文件json
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    #初始化时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    #初始化一个字典，保存环境和时间种子等关键信息
    meta = dict()
    env_info_dict = collect_env()
    #记录一些环境信息
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info，记录基础信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    #设置随机数种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        #print('111')
        cfg.checkpoint_config.meta = dict(
            clsnet_version = '0.0.1',
            config = cfg.pretty_text,
            CLASSES =datasets[0].CLASSES
        )
        #print(cfg.checkpoint_config.meat)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device='cpu', #if args.device == 'cpu' else 'cuda',
        meta=meta)


if __name__=='__main__':
    main()
