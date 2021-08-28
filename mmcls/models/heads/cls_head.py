from mmcls.models.heads import BaseHead
from mmcls.models.builder import HEADS,build_loss
from mmcls.models.loss import Accuracy


import torch.nn as nn

@HEADS.register_module()
class ClsHead(BaseHead):

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss',loss_weight=1.0),
                 topk=(1,),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead,self).__init__(init_cfg=init_cfg)
        
        assert isinstance(loss,dict)
        assert isinstance(topk,(int,tuple))
        if isinstance(topk, int):
            topk = (topk,)
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc