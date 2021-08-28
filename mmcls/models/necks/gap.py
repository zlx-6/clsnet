from typing import NewType
import torch
import torch.nn as nn

from mmcls.models.builder import NECKS

@NECKS.register_module()
class GlobalAveragePooling(nn.Module):

    def __init__(self,dim=1):
        super(GlobalAveragePooling,self).__init__()
        #现在只支持2d的池化
        self.gap = nn.AdaptiveAvgPool2d((1,1))
