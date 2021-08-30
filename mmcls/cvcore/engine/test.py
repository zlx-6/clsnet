import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

from mmcls.cvcore.runner import get_dist_info
from mmcls.cvcore.utils import ProgressBar


def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results