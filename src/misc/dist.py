"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

by lyuwenyu
"""

import time 
import os
import random
import numpy as np 

import torch
import torch.nn as nn 
import torch.distributed
import torch.distributed as tdist

from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def init_distributed():
    '''
    distributed setup
    args:
        backend (str), ('nccl', 'gloo')
    '''
    try:
        # # https://pytorch.org/docs/stable/elastic/run.html
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        # RANK = int(os.getenv('RANK', -1))
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        tdist.init_process_group()
        torch.distributed.barrier()

        rank = get_rank()
        device = torch.device(f'cuda:{rank}')
        # setting default device
        torch.cuda.set_device(device)

        setup_print(rank == 0)
        print('Initialized distributed mode...')

        return True 

    except Exception as e:
        print(e)
        print('Not init distributed mode.')
        return False 


def setup_print(is_main):
    '''This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    return tdist.is_available() and tdist.is_initialized()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, file_path):
    if is_main_process():
        save_dir = os.path.dirname(file_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(obj, file_path)


def warp_model(model, find_unused_parameters=False, sync_bn=False,):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model 
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    return model


def warp_loader(loader):
    if is_dist_available_and_initialized():
        device_count = torch.cuda.device_count()
        dataset = loader.dataset
        sampler = DistributedSampler(dataset, shuffle=loader.shuffle)
        loader = DataLoader(dataset, 
                            loader.batch_size // device_count, 
                            sampler=sampler, 
                            drop_last=loader.drop_last, 
                            collate_fn=loader.collate_fn, 
                            pin_memory=loader.pin_memory,
                            num_workers=(loader.num_workers + device_count - 1) // device_count)
        
    
    return loader


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    '''
    Args 
        data dict: input, {k: v, ...}
        avg bool: true
    '''
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size
        
        _data = {k: v for k, v in zip(keys, values)}
    
    return _data


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list

    
def sync_time():
    '''sync_time
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def set_seed(seed):
    # fix the seed for reproducibility
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


