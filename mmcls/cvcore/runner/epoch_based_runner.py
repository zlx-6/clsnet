import os.path as osp
import platform
import shutil
import time
import warnings
import torch

from mmcls.cvcore.runner import BaseRunner,RUNNERS,save_checkpoint,get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    
    def run_iter(self):
        return 0