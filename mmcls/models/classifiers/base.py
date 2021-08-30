from re import L
from typing import OrderedDict
from mmcls.cvcore.runner import BaseModule

import cv2
import torch
import torch.nn as nn
import warnings
from abc import ABCMeta,abstractmethod
import torch.distributed as dist


class BaseClassifier(BaseModule,metaclass=ABCMeta):

    def __init__(self, init_cfg):
        super(BaseClassifier,self).__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
    
    @property
    def with_neck(self):
        return hasattr(self,'neck') and self.neck is not None
    
    @property
    def with_head(self):
        return hasattr(self,'head') and self.head is not None

    @abstractmethod
    def extract_feat(self,imgs):
        pass

    def extract_feats(self,imgs):
        assert isinstance(imgs,list)
        for img in imgs:
            yield self.extract_feat(img)
    
    def forward_train(self,imgs,**kwargs):

        pass

    #@abstractmethod
    def simple_test(self,imgs,**kwargs):
        pass
    
    def forward_test(self,imgs,**kwargs):
        
        if isinstance(imgs,torch.Tensor):
            imgs = [imgs]
        for var,name in [(imgs,'img')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    #@auto_fp16(apply_to=('img',))
    def forward(self,img,return_loss=True,**kwargs):
        if return_loss:
            return self.forward_train(img,**kwargs)
        else:
            return self.forward_test(img,**kwargs)

    def _parse_losses(self,losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self,data,optimizer):
        
        loss=self(**data)

        loss, log_vars = self._parse_losses(loss)

        outputs = dict(loss=loss,log_vars=log_vars,num_samples=len(data['img'].data)) 
    
        return outputs
