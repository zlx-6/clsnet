from abc import ABCMeta,abstractmethod

from mmcls.cvcore.runner import BaseModule

class BaseBackbone(BaseModule,metaclass=ABCMeta):

    def __init__(self, init_cfg=None):
        super(BaseBackbone,self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)
    