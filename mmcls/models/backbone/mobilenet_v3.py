from mmcls.models import BACKBONES 
from mmcls.models.backbone import BaseBackbone



@BACKBONES.register_module()
class MobileNetV3(BaseBackbone):
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 2],
                  [3, 88, 24, False, 'ReLU', 1],
                  [5, 96, 40, True, 'HSwish', 2],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 120, 48, True, 'HSwish', 1],
                  [5, 144, 48, True, 'HSwish', 1],
                  [5, 288, 96, True, 'HSwish', 2],
                  [5, 576, 96, True, 'HSwish', 1],
                  [5, 576, 96, True, 'HSwish', 1]],
        'large': [[3, 16, 16, False, 'ReLU', 1],
                  [3, 64, 24, False, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 1],
                  [5, 72, 40, True, 'ReLU', 2],
                  [5, 120, 40, True, 'ReLU', 1],
                  [5, 120, 40, True, 'ReLU', 1],
                  [3, 240, 80, False, 'HSwish', 2],
                  [3, 200, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 480, 112, True, 'HSwish', 1],
                  [3, 672, 112, True, 'HSwish', 1],
                  [5, 672, 160, True, 'HSwish', 2],
                  [5, 960, 160, True, 'HSwish', 1],
                  [5, 960, 160, True, 'HSwish', 1]]
    } 

    def __init__(self,
                arch='small',
                conv_cfg=None,
                norm_cfg=dict(type ='BN',eps=0.001,momentum=0.01),
                out_indices=None,
                frozen_stages=-1,
                norm_eval=False,
                with_cp=False,
                init_cfg = [
                    dict(
                        type='Kaiming',
                        layer=['Conv2d'],
                        nolinearity='leaky_relu'),
                    dict(type='Normal',layer=['Linear'],std=0.01),
                    dict(type='Constant',layer=['BatchNorm2d'],val=1)
                ]):
        super(MobileNetV3,self).__init__(init_cfg)
        assert arch in self.arch_settings
        if out_indices is None:
            out_indices = (12,) if arch == 'small' else (16,)
        for order,index in enumerate(out_indices):
            if index not in range(0,len(self.arch_settings[arch])+2):
                raise ValueError(f'the item in out_indices must in the range (0, {len(self.arch_settings[arch])+2}, but received {index}')

        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = self._make_layer()
        self.feat_dim = self.arch_settings[arch][-1][1]

    def _make_layer(self):
        pass