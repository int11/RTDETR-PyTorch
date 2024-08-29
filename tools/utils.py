import os
import sys
import time
import math
import datetime
from typing import Iterable

import torch
from torch.cuda.amp import GradScaler
import torch.optim.lr_scheduler as lr_schedulers
import torch.amp 

from src.zoo import rtdetr_train_dataloader, rtdetr_val_dataloader, rtdetr_criterion
from src.data.coco.coco_eval import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.misc import MetricLogger, SmoothedValue, reduce_dict
from src.optim.ema import ModelEMA
from src.nn.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.nn.rtdetr.utils import *
import src.misc.dist as dist


def fit(model, 
        weight_path, 
        optimizer, 
        save_dir,
        criterion=None,
        train_dataloader=None, 
        val_dataloader=None,
        epoch=73,
        use_amp=True,
        use_ema=True):

    if criterion == None:
        criterion = rtdetr_criterion()
    if train_dataloader == None:
        train_dataloader = rtdetr_train_dataloader()
    if val_dataloader == None:
        val_dataloader = rtdetr_val_dataloader()


    scaler = GradScaler() if use_amp == True else None
    ema_model = ModelEMA(model, decay=0.9999, warmups=2000) if use_ema == True else None
    lr_scheduler = lr_schedulers.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1) 

    last_epoch = 0
    if weight_path != None:
        last_epoch = load_tuning_state(weight_path, model, ema_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ema_model.to(device) if use_ema == True else None
    criterion.to(device)  
    
    #dist wrap modeln loader must do after model.to(device)
    if dist.is_dist_available_and_initialized():
        # model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)
        # ema_model = dist.warp_model(ema_model, find_unused_parameters=True, sync_bn=True) if use_ema == True else None
        # criterion = dist.warp_model(criterion, find_unused_parameters=True, sync_bn=True)
        train_dataloader = dist.warp_loader(train_dataloader)
        val_dataloader = dist.warp_loader(val_dataloader)
        model = dist.warp_model(model, find_unused_parameters=False, sync_bn=True)

    
    print("Start training")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    start_time = time.time()
    
    for epoch in range(last_epoch + 1, epoch):
        if dist.is_dist_available_and_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        train_one_epoch(model, criterion, train_dataloader, optimizer, device, epoch, max_norm=0.1, print_freq=100, ema=ema_model, scaler=scaler)

        lr_scheduler.step()

        dist.save_on_master(state_dict(epoch, model, ema_model), os.path.join(save_dir, f'{epoch}.pth'))

        module = ema_model.module if use_ema == True else model
        test_stats, coco_evaluator = val(model=module, weight_path=None, criterion=criterion, val_dataloader=val_dataloader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    metric_logger = MetricLogger(data_loader, header=f'Epoch: [{epoch}]')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, cache_enabled=True):
            outputs = model(samples, targets)
        

        with torch.autocast(device_type=device.type, enabled=False):
            loss_dict = criterion(outputs, targets)

        loss = sum(loss_dict.values())


        if scaler is not None:
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
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


@torch.no_grad()
def val(model, weight_path, criterion=None, val_dataloader=None):
    if criterion == None:
        criterion = rtdetr_criterion()
    if val_dataloader == None:
        val_dataloader = rtdetr_val_dataloader()

    model.eval()
    criterion.eval()

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    postprocessor = RTDETRPostProcessor(num_top_queries=300, remap_mscoco_category=val_dataloader.dataset.remap_mscoco_category)
    iou_types = postprocessor.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)


    if weight_path != None:
        state = torch.hub.load_state_dict_from_url(weight_path, map_location='cpu') if 'http' in weight_path else torch.load(weight_path, map_location='cpu')
        if 'ema' in state:
            model.load_state_dict(state['ema']['module'], strict=False)
        else:
            model.load_state_dict(state['model'], strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(val_dataloader, header='Test:',)

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

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
