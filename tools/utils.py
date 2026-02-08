"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import argparse
import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.data.coco.coco_eval import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.misc import MetricLogger, SmoothedValue, reduce_dict

import src.misc.dist_utils as dist_utils


def fit(model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        save_dir: str,
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        criterion: torch.nn.Module,
        lr_scheduler,
        postprocessor: torch.nn.Module,
        ema_model = None,
        scaler = None,
        train_epoch: int = 73,
        resume_epoch: int = 0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if ema_model != None: ema_model.to(device) 
    criterion.to(device)  
    
    #dist wrap model loader must do after model.to(device)
    if dist_utils.is_dist_available_and_initialized():
        train_dataloader = dist_utils.warp_loader(train_dataloader)
        val_dataloader = dist_utils.warp_loader(val_dataloader)
        model = dist_utils.warp_model(model, find_unused_parameters=False, sync_bn=True)

    
    print("Start training")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    start_time = time.time()
    

    for train_epoch in range(resume_epoch + 1, train_epoch):
        # set dataloader epoch parameter
        train_dataloader.sampler.set_epoch(train_epoch) if dist_utils.is_dist_available_and_initialized() else train_dataloader.set_epoch(train_epoch)
        
        train_one_epoch(model, criterion, train_dataloader, optimizer, device, train_epoch, max_norm=0.1, print_freq=100, ema=ema_model, scaler=scaler)

        lr_scheduler.step()

        dist_utils.save_on_master(state_dict(train_epoch, model, ema_model, optimizer, lr_scheduler, scaler), os.path.join(save_dir, f'{train_epoch}.pth'))

        module = ema_model.module if ema_model != None else model
        test_stats, coco_evaluator = val(model=module, criterion=criterion, val_dataloader=val_dataloader, postprocessor=postprocessor)
        
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    print_freq: int,
                    max_norm: float = 0,
                    ema=None, 
                    scaler=None):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(data_loader, header=f'Epoch: [{epoch}]', print_freq=print_freq)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, cache_enabled=True, enabled=scaler != None and device.type == 'cuda'):
            outputs = model(samples, targets)
        
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())

        #amp
        if scaler != None:
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()
        
        # ema 
        if ema != None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


#TODO This function too complex and slow because it from original repository, need to refactor
@torch.no_grad()
def val(model: torch.nn.Module, 
        val_dataloader: DataLoader, 
        criterion: torch.nn.Module, 
        postprocessor: torch.nn.Module, 
        scaler=None):
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    if dist_utils.is_dist_available_and_initialized():
        val_dataloader = dist_utils.warp_loader(val_dataloader)
        model = dist_utils.warp_model(model, find_unused_parameters=False, sync_bn=True)
    
    model.eval()
    criterion.eval()

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    coco_evaluator = CocoEvaluator(base_ds, ['bbox'])
    iou_types = coco_evaluator.iou_types
    
    metric_logger = MetricLogger(val_dataloader, header='Test:',)

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=scaler != None and device.type == 'cuda'):
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'a')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self  # Redirect stdout to this instance
        print(f"===== Logging session started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        return self

    def write(self, obj):
        self.file.write(obj)
        self.file.flush()
        self.stdout.write(obj)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"===== Logging session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = self.stdout  # Restore original stdout
        self.file.close()


def load_tuning_state(path, model, ema_model=None, optimizer=None, lr_scheduler=None, scaler=None):
    """only load model for tuning and skip missed/dismatched keys
    """
    state = torch.hub.load_state_dict_from_url(path, map_location='cpu') if 'http' in path else torch.load(path, map_location='cpu')

    infos = dist_utils.de_parallel(model).load_state_dict(state['model'], strict=False)
    print(f'Load model.state_dict, {infos}')

    if 'ema' in state:
        if ema_model is None:
            raise RuntimeError('WARNING, ema model weight exist in file but flag is use_ema=False')
        else:
            infos = ema_model.load_state_dict(state['ema'], strict=False)
            print(f'Load ema_model.state_dict, {infos}')

    if 'optimizer' in state and optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        print(f'Load optimizer.state_dict')

    if 'lr_scheduler' in state and lr_scheduler is not None:
        lr_scheduler.load_state_dict(state['lr_scheduler'])
        print(f'Load lr_scheduler.state_dict')

    if 'scaler' in state and scaler is not None:
        scaler.load_state_dict(state['scaler'])
        print(f'Load scaler.state_dict')

    return state['last_epoch']


def state_dict(last_epoch, model, ema_model=None, optimizer=None, lr_scheduler=None, scaler=None):
    '''current train info state dict 
    '''
    state = {}
    state['model'] = dist_utils.de_parallel(model).state_dict()
    state['date'] = datetime.now().isoformat()
    state['last_epoch'] = last_epoch

    if ema_model is not None:
        state['ema'] = ema_model.state_dict()

    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()

    if lr_scheduler is not None:
        state['lr_scheduler'] = lr_scheduler.state_dict()

    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    return state