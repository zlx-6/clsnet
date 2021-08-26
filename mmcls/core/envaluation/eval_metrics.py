from numbers import Number

import numpy as np
from numpy.lib.function_base import average
import torch

def calculate_confusion_matrix(pred,target):

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _,pred_label = pred.topk(1,dim=1)
    pred_label = pred_label.view(-1)
    target_label =target.view(-1)
    confusion_matrix = torch.zeros(num_classes,num_classes)

    with torch.no_grad:
        for t,p in zip(target_label,pred_label):
            confusion_matrix[t.long(),p.long()]+=1

    return confusion_matrix

def support(pred, target,average_mode='macro'):

    confusion_matrix = calculate_confusion_matrix(pred,target)
    with torch.no_grad():
        #计算每个类别有多少张
        res = confusion_matrix.sum(1)
        if average_mode =='macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        
    return res

