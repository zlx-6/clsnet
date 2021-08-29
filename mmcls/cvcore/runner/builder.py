from mmcls.cvcore.registry import build_from_cfg,Registry

RUNNERS = Registry('runner')

def build_runner(cfg,default_args=None):
    return build_from_cfg(cfg,RUNNERS,default_args=default_args)