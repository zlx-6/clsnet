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

def precision_recall_f1(pred,target,average_mode='macro',thrs = 0.):
    allowed_average_mode = ['macro','none']

    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsuporrt type of averaging {average_mode}.')

    if isinstance(pred,torch.Tensor):
        pred = pred.numpy()
    if isinstance(target,torch.Tensor):
        target = target.numpy()

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    #取最大的值的label
    pred_label = np.argsort(pred,axis=1)[:,-1]
    pred_socre = np.sort(pred,axis=1)[:,-1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        _pred_label = pred_label.copy
        if thr is not None:
            _pred_label[pred_socre<=thr] = -1#概率小于thr的不会属于那一类
        pred_positive = label == _pred_label.reshape(-1,1)
        gt_positive = label == target.reshape(-1,1)
        precision = (pred_positive & gt_positive).sum(0)/np.maximum(pred_positive.sum(0),1)*100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall, 1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores
