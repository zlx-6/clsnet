from abc import ABCMeta,abstractmethod

from mmcls.cvcore.runner import BaseModule

class BaseBackbone(BaseModule,metaclass=ABCMeta):

    def __init__(self, init_cfg=None):
        super(BaseBackbone,self).__init__(init_cfg=init_cfg)
    