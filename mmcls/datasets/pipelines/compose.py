from mmcls.cvcore import build_from_cfg
from mmcls.datasets.builder import PIPELINES

@PIPELINES.register_module()
class Compose(object):

    def __init__(self,transforms):
        self.transforms = []
       
        for transform in transforms:
            if isinstance(transform,dict):
                transform=build_from_cfg(transform,PIPELINES)#第二个参数是从哪个Register里面取dict_modules，Register下会注册许多modules
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)


    def __call__(self,data):
        #print(self.transforms)
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string +=f'\n {t}'
        format_string += '\n)'
        return format_string#打印用的

if __name__ == "__main__":
    print(Compose(None))