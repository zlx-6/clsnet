import copy
import warnings

from torch.functional import norm

from mmcls.models.classifiers import BaseClassifier
from mmcls.models.utils import Augments
from mmcls.models.builder import CLASSIFIERS,build_backbone,build_neck,build_head,build_loss

@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):
    
    def __init__(self,
                backbone,
                neck = None,
                head = None,
                pretrained = None,
                train_cfg = None,
                init_cfg=None):
        super(ImageClassifier,self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained',checkpoint=pretrained)

        self.backbone = build_backbone(backbone)#mobilenetv3

        if neck is not None:
            self.neck = build_neck(neck)
        if head is not None:
            self.head = build_head(head)

        self.augments = None
        #一些增强方法？？？,先不实现
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments',None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)
    def extract_feat(self,img):
        imgs=1
        return imgs

