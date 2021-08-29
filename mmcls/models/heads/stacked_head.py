from mmcls.models.heads import BaseHead,ClsHead
from mmcls.models.builder import HEADS

import torch.nn as nn
from typing import Dict, Sequence
from mmcls.cvcore.runner import ModuleList,BaseModule
from mmcls.cvcore.cnn import build_norm_layer,build_activation_layer

class LinearBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

@HEADS.register_module()
class StackedLinearClsHead(ClsHead):
    
    def __init__(self, num_classes,in_channels,mid_channels,dropout_rate=0.1,
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
        self.layers = ModuleList(
            init_cfg=dict(
                type='Normal', layer='Linear', mean=0., std=0.01, bias=0.))
        in_channels =self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.,
                norm_cfg=None,
                act_cfg=None))
    def init_weights(self):
        self.layers.init_weights()
