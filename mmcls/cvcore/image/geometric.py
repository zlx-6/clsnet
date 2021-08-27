import cv2
import numbers
import numpy as np

def bbox_clip(bboxes,img_shape):
    cmin = np.empty(bboxes.shape[-1],dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes

def bbox_scaling(bboxes,scale,clip_shape=None):
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(img,bboxes,scales = 1.0 ,pad_fill = None):
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill,(int,float)):
            pad_fill = [pad_fill for _ in range(chn)]
    
    _bboxes = bboxes[None,...] if bboxes.ndim == 1 else bboxes#array([[ 4,  0, 35, 31]])
    scaled_bboxes = bbox_scaling(_bboxes,scales).astype(np.int32)
    clipped_bboxes = bbox_clip(scaled_bboxes,img.shape)#这俩函数可以合并吧array([ 7,  8, 38, 39])
    
    patches =[]
    for i in range(clipped_bboxes.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bboxes[i,:])
        if pad_fill is None:
            patch = img[y1:y2+1,x1:x2+1,...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 1:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(img,*,shape=None,padding=None,pad_val=0,padding_mode='constant'):
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0,0,shape[1]-img.shape[1],shape[0]-img.shape[0])
    
    if isinstance(pad_val,tuple):
        assert len(pad_val) == img.shape[-1]#每个通道吗？？

    if isinstance(padding,tuple) and len(padding) in [2,4]:
        if len(padding)==2:
            padding == (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):#检查是不是一个数,包括浮点数和小数
        padding = (padding, padding, padding, padding)#(4, 4, 4, 4)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')
    
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(img,padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)#(32, 32, 3)-->(40, 40, 3)

    return img

def imflip(img,direction='horizontal'):

    assert direction in ['horizontal','vertical','duagonal']
    if direction == 'horizontal':
        return np.flip(img,axis = 1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))
