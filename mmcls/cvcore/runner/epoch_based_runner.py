import enum

from torch.utils.data.dataset import T
from mmcls.apis.train import train_model
import os.path as osp
import platform
import shutil
import time
import warnings
import torch

from mmcls.cvcore.utils.misc import is_list_of
from mmcls.cvcore.runner import BaseRunner,RUNNERS,save_checkpoint,get_host_info
from mmcls.cvcore.utils import symlink

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    
    def run_iter(self,data_batch,train_mode,**kwargs):
        if self.batch_processor is not None:
             outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch,self.optimizer,**kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        
    def train(self,data_loader,**kwargs):
        self.model.train()#https://www.zhihu.com/question/429337764/answer/1564983277
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')#得到第一次epoch的学习率
        time.sleep(2)
        for i,data_batch in enumerate(self.data_loader):
            self._inner_iter = i#data_batch:{img_meta:DC(filp,filp_directions,img_norm,std,..),img:torch.Size([16, 3, 32, 32]),gt_label:torch.Size([16])}
            self.call_hook('before_train_iter')#得到这次迭代的学习率，warmup
            self.run_iter(data_batch,train_mode=True,**kwargs)
            self.call_hook('after_train_iter')#更新参数等
            self._iter += 1

        self.call_hook('after_train_epoch')#保存pth文件
        self._epoch += 1
    
    @torch.no_grad()
    def val(self,data_loader,**kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')#啥也没做
        time.sleep(2)
        for i,data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')#啥也没做
            self.run_iter(data_batch,train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self,data_loaders,workflow,max_epochs=None,**kwargs):
        assert isinstance(data_loaders,list)
        assert is_list_of(workflow,tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')#设置学习率，log等的保存路径等hook

        while self.epoch < self._max_epochs:
            for i,flow in enumerate(workflow):
                mode, epochs = flow
                #mode='val'#这行要删掉
                if isinstance(mode,str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))
                
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i],**kwargs)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)#epoch_1.pth
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)