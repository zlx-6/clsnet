import torch.nn as nn

from mmcls.cvcore.cnn.bricks import ConvModule
from mmcls.cvcore.runner import BaseModule
from mmcls.models.utils import make_divisible

class SELayer(BaseModule):
    
    def __init__(self,
                channels,
                squeeze_channels=None,
                ratio=16,#4,中间层的缩小的倍数
                divisor=8,
                bias='auto',
                conv_cfg=None,
                act_cfg=(dict(type='ReLU'),dict(type='Sigmoid')),
                init_cfg=None):
        super(SELayer,self).__init__(init_cfg)
        if isinstance(act_cfg,dict):
            act_cfg = (act_cfg,act_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if squeeze_channels is None:
            squeeze_channels = make_divisible(channels//ratio,divisor)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out