from mmcls.models.utils.helpers import is_tracing
from mmcls.models.heads import BaseHead
from mmcls.models.builder import HEADS,build_loss
from mmcls.models.loss import Accuracy
from mmcls.models.utils import is_tracing

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self,cls_score):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        return self.post_process(pred)

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
        
