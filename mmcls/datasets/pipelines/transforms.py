from mmcv.image.geometric import imcrop
from mmcls.datasets.builder import PIPELINES
from mmcls.cvcore.image import impad,imcrop,imflip

import numpy as np
import random
#import mmcv

@PIPELINES.register_module()
class RandomCrop(object):

    def __init__(self,size,padding=None,pad_if_needed=False,pad_val=0,padding_module ='constant'):
        if isinstance(size,(tuple,list)):
            self.size = size
        else:
            self.size = (size,size)#32
        self.padding = padding #4
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val#0
        self.padding_module = padding_module

    @staticmethod
    def get_params(img,output_size):

        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return ymin, xmin, target_height, target_width


    def __call__(self,results):
        for key in results.get('img_fileds',['img']):
            img = results[key]
            if self.padding is not None:
                img = impad(img,padding = self.padding,padding_mode=self.padding_module)
            #填充高度
            if self.pad_if_needed and img.shape[0]<self.size[0]:
                img = impad(img,padding=(0, self.size[0] - img.shape[0], 0,#左上右下模式,这个填充模式有错吗？？？
                             self.size[0] - img.shape[0]),pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = impad(
                    img,
                    padding=(self.size[1] - img.shape[1], 0,
                             self.size[1] - img.shape[1], 0),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            ymin,xmin,height,width = self.get_params(img,self.size)#得到随机剪裁的目标#0,4,32,32
            results[key] = imcrop(img,np.array([
                    xmin,
                    ymin,
                    xmin + width - 1,
                    ymin + height - 1,
                ]))
        return results

    def __repr__(self):
        return (self.__class__.__name__ + f'(size = {self.size}, padding = {self.padding})')

@PIPELINES.register_module()
class RandomFlip(object):

    def __init__(self,flip_prob=0.5,direction='horizontal') -> None:
        assert 0<=flip_prob<=1
        assert direction in ['horizontal','vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self,results):
        
        flip =True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            for key in results.get('img_fields',['img']):
                results[key] = imflip(results[key],direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__+f'flip_prob = {self.flip_prob}'