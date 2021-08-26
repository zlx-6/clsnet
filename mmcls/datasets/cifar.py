from mmcls.datasets.builder import DATASETS
from mmcls.datasets.base_datasets import BaseDataset
from mmcls.cvcore import get_dist_info
from mmcls.datasets.utils import check_integrity,download_and_extract_archive

import os
import torch.distributed as dist
import pickle
import numpy as np

@DATASETS.register_module()
class CIFAR10(BaseDataset):
    base_folder = 'cifar-10-batches-py'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def load_annotations(self):
        rank,world_size = get_dist_info()#单卡训练时返回0,1
        #print(rank,world_size)
        if rank ==0 and not self._check_inteck_integrity():#如果文件没有下载，会自动下载,下载了会检查md5
             # download_and_extract_archive是下载并提取文件的函数
             download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)
        if world_size >1:
            dist.barrier()
        
        if not self.test_mode:
            download_list = self.train_list
        else:
            download_list = self.test_list
        
        self.imgs = []
        self.gt_labels = []

        for file_name,check_sum in download_list:
            file_path = os.path.join(self.data_prefix,self.base_folder,file_name)
            with open(file_path,'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.imgs.append(entry['data'])
                if 'labels' in entry:
                    self.gt_labels.append(entry['labels'])
                else:
                    self.gt_labels.extend(entry['fine_labels'])  
        self.imgs = np.vstack(self.imgs).reshape(-1,3,32,32)  
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    def _load_meta(self):
        path = os.path.join(self.data_prefix, self.base_folder,
                            self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.CLASSES = data[self.meta['key']]

    def _check_inteck_integrity(self):
        root = self.data_prefix
        for fentry in (self.train_list + self.test_list):
            file_name, md5 = fentry[0],fentry[1]
            fpath = os.path.join(root,self.base_folder,file_name)
            if not check_integrity(fpath,md5):
                return False
        return True
    


if __name__ =="__main__":
    CIFAR10('data\cifar10',None,None)