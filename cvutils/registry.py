import inspect

def build_from_cfg(cfg,registry,default_args=None):
    
    if not isinstance(cfg,dict):
        raise TypeError(f'cfg must be a dict, but get a {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(f'"cfg" or "default_args" must cotain the key "type", but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be an Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}') 
    
    args = cfg.copy()   

    if default_args is not None:
        for name,value in default_args.items():
            args.setdefault(name,value)
    
    obj_type = args.pop('type')
    if isinstance(obj_type,str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}:{e}')
    

class Registry:

    def __init__(self,name,build_func=None, parent=None,scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope #??在哪个函数下？？

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        
        if parent is not None:
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = parent 

    def __len__(self):
        return len(self._module_dict)  

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.


        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__#'__main__'
        split_filename = filename.split('.')#['__main__']
        return split_filename[0]#'__main__'

    def get(self,key):

        scope, real_key = self.split_score_key(key)

        if scope is None or scope == self._scope:
            if real_key in self.__module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key) 

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    def _register_module(self, module_class, module_name = None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name,str):
            module_name = [module_name]
        
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def register_module(self,name=None,force=None,module = None):

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register

if __name__ == "__main__":
    Registry('reg')
