from abc import ABCMeta
from mmcls.cvcore import build_from_cfg,Registry

AUGMENT = Registry('augments')

def build_augment(cfg,default_args=None):
    return build_from_cfg(cfg,AUGMENT,default_args)