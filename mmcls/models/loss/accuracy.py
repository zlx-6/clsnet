from mmcls import cvcore
import torch
import numpy as np
from numbers import Number

def accuracy_torch(pred,target,topk,thrs):
    if isinstance(thrs,Number):
        thrs = (thrs,)
        res_signle =True
    elif isinstance(thrs,tuple):
        res_signle = False
    
    res = []
    maxk=max(topk)
    num= pred.size(0)
    pred_score,pred_label = pred.topk(maxk,dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1,-1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            #只有大于阈值的作为正确的
            _correct = correct & (pred_score.t()>thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0,keepdim = True)#top5时只要有一个对的就是对的
            res_thr.append(correct_k.mul_(100/num))
        if res_signle:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    
    return res
    

def accuracy(pred,target,topk=1,thrs=.0):

    assert isinstance(topk,(int,tuple))

    if isinstance(topk,int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False
    #现在只考虑tensor类型的预测
    if isinstance(pred, torch.Tensor) and isinstance(target,torch.Tensor):
        res = accuracy_torch(pred,target,topk,thrs)
    
    return res[0] if return_single else res
    