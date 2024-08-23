"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 
import time
from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


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

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
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