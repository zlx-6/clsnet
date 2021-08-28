from mmcls.models.heads import BaseHead,ClsHead
from mmcls.models.builder import HEADS

import torch.nn as nn
from typing import Dict, Sequence


@HEADS.register_module()
class StackedLinearClsHead(ClsHead):
    
    def __init__(self, num_classes,in_channels,mid_channels,dropout_rate =0.1,
                 norm_cfg=None,act_cfg=dict(type = 'ReLU'),**kwargs ):
        super(StackedLinearClsHead,self).__init__(**kwargs)
        assert num_classes > 0, \
            f'`num_classes` of StackedLinearClsHead must be a positive ' \
            f'integer, got {num_classes} instead.'
        self.num_classes = num_classes

        self.in_channels = in_channels

        assert isinstance(mid_channels, Sequence), \
            f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
            f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()

    def _init_layers(self):
        pass

