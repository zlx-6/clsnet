from mmcls.cvcore.runner import BaseModule

import cv2
import torch
import torch.nn as nn
import warnings
from abc import ABCMeta,abstractmethod


class BaseClassifier(BaseModule,metaclass=ABCMeta):

    def __init__(self, init_cfg):
        super(BaseClassifier,self).__init__(init_cfg=init_cfg)
        self.fp16_enabled = False

    