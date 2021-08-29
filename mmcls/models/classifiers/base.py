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
    
    @property
    def with_neck(self):
        return hasattr(self,'neck') and self.neck is not None
    
    @property
    def with_head(self):
        return hasattr(self,'head') and self.head is not None

    @abstractmethod
    def extract_feat(self,imgs):
        pass

    def extract_feats(self,imgs):
        assert isinstance(imgs,list)
        for img in imgs:
            yield self.extract_feat(img)
    
        