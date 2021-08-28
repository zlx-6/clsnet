from mmcls.cvcore.registry import Registry
from mmcls.cvcore.cnn.builder import MODELS
from mmcls.cvcore.cnn import MODELS as CV_MODELS

MODELS = Registry('models',parent = CV_MODELS)


CLASSIFIERS = MODELS
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)