from numbers import Number

import numpy as np
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


    
