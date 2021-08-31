from abc import ABCMeta,abstractmethod
import torch.nn.functional as F

from mmcls.cvcore.runner import BaseModule

class BaseHead(BaseModule,metaclass = ABCMeta):

    def __init__(self,init_cfg=None):
        super(BaseHead,self).__init__(init_cfg)
    
    @abstractmethod
    def forward_train(self, x, gt_label, **kwargs):
        pass

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

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = img
        for layer in self.layers:
            cls_score = layer(cls_score)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)
    
    