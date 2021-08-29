import torch.nn as nn

from mmcls.cvcore.cnn.bricks import ACTIVATION_LAYERS

@ACTIVATION_LAYERS.register_module()
class HSwish(nn.Module):

    def __init__(self,inplace=False):
        super(HSwish,self).__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self,x):
        return x*self.act(x+3)/6