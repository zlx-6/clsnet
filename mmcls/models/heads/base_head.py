from abc import ABCMeta,abstractmethod

from mmcls.cvcore.runner import BaseModule

class BaseHead(BaseModule,metaclass = ABCMeta):

    def __init__(self,init_cfg=None):
        super(BaseHead,self).__init__(init_cfg)
    
    