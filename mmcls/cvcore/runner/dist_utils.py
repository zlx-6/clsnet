import torch
import functools
from collections import OrderedDict
from torch import distributed as dist
from torch.random import initial_seed
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

TORCH_VERSION =torch.__version__

def get_dist_info():
    if TORCH_VERSION < '1.0':
        initialized = dist._initialized

    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        rank,_ = get_dist_info()
        if rank == 0:
            return func(*args,**kwargs)
    
    return wrapper

def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce parameters.

    Args:
        params (list[torch.Parameters]): List of parameters or buffers of a
            model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))

def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)