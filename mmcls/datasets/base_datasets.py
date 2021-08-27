#from numpy.lib.function_base import average
import torch
import torch as nn
from torch.utils.data import Dataset
import copy
import numpy as np

from abc import ABCMeta, abstractclassmethod  # 这个不知道是啥
from mmcls.datasets.pipelines import Compose
from mmcls.cvcore.fileio import list_from_file
from mmcls.models.loss import accuracy
from mmcls.core.envaluation import support,precision_recall_f1

class BaseDataset(Dataset, metaclass=ABCMeta):
    CLASSES = None

    def __init__(self, data_prefix, pipeline, classes, ann_file=None, test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix#'data\\cifar10'
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)#None
        self.data_infos = self.load_annotations()#{img,}

    
    @abstractclassmethod
    def load_annotations(self):#具体数据集需要实现
        pass

    def __getitem__(self, index):
       
       return self.prepare_data(index)

    def __len__(self):
        return len(self.data_infos)

    def prepare_data(self,idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)


    @classmethod
    def get_classes(cls,classes=None):
        if classes is None:
            return cls.CLASSES
        
        #如果是个路径
        if isinstance(classes,str):
            class_names = list_from_file(classes)
        elif isinstance(classes,(tuple,list)):
            class_names = classes

        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_gt_labels(self):

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def evaluate(self,results,metric='accuracy',metric_options=None,logger=None):

        if metric_options is None:
            metric_options = {'topk':(1,5)}
        if isinstance(metric,str):
            metric_options = [metric]
        else:
            metrics = metric

        allowed_metrics = [
            'accuracy','precision','recall','f1_score','support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        
        invalid_metrics = set(metric) - set(allowed_metrics)
        if len(invalid_metrics) !=0:
            raise ValueError(f'metric {invalid_metrics} is not suppported')
        
        topk = metric_options.get('topk',(1,5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode','macro')

        if 'accuray' in metrics:
            if thrs is not None:
                acc = accuracy(results,gt_labels,topk,thrs)
            else:
                acc =accuracy(results,gt_labels,topk)
            if isinstance(topk,tuple):
                eval_results_ = {
                    f'accuracy_top-{k}':a
                    for k,a in zip(topk,acc)
                }
            else:
                eval_results_ = {'accuracy':acc}
            if isinstance(thrs,tuple):
                for key,values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs,values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_val = support(results,gt_labels,average_mode=average_mode)
            eval_results['support'] = support_val
            
        precision_recall_f1_keys = ['precision','recall','f1_score']
        if len(set(metrics)&set(precision_recall_f1_keys))!=0:
            if thrs is not None:
               precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
