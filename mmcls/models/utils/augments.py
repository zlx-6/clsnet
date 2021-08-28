import random
import numpy as np

from mmcls.models.utils import build_augment

class Augments(object):

    def __init__(self,augments_cfg) :
        super(Augments,self).__init__()

        if isinstance(augments_cfg,dict):
            augments_cfg = [augments_cfg]

        self.augments = [build_augment(cfg) for cfg in augments_cfg]
        self.augments_probs = [aug.prob for aug in self.augments]#具体的增强先不做实现

        has_identity = any([cfg['type'] == 'Identity' for cfg in augments_cfg])
        if has_identity:
            assert sum(self.augment_probs) == 1.0,\
                'The sum of augmentation probabilities should equal to 1,' \
                ' but got {:.2f}'.format(sum(self.augment_probs))
        else:
            assert sum(self.augment_probs) <= 1.0,\
                'The sum of augmentation probabilities should less than or ' \
                'equal to 1, but got {:.2f}'.format(sum(self.augment_probs))
            identity_prob = 1 - sum(self.augment_probs)
            if identity_prob > 0:
                num_classes = self.augments[0].num_classes
                self.augments += [
                    build_augment(
                        dict(
                            type='Identity',
                            num_classes=num_classes,
                            prob=identity_prob))
                ]#Identity也暂时不做实现
                self.augment_probs += [identity_prob]