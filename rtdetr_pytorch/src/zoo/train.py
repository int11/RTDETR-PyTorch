import os
import time
import datetime
import json


from torch.cuda.amp import GradScaler
from src.zoo.dataloader import rtdetr_train_dataloader, rtdetr_val_dataloader
from src.zoo.criterion import rtdetr_criterion
from src.data.coco.coco_eval import CocoEvaluator
from src.misc.logger import MetricLogger
from src.solver.det_engine import train_one_epoch
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.optim.optim import AdamW
from src.optim.ema import ModelEMA

from src.nn.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.nn.rtdetr import rtdetr
from src.nn.rtdetr.utils import *

import src.misc.dist as dist
import torch.optim.lr_scheduler as lr_schedulers

def fit(model, 
        weight_path, 
        optimizer, 
        save_dir,
        criterion=None,
        train_dataloader=None, 
        val_dataloader=None,
        epoch=72,
        use_amp=True,
        use_ema=True):

    if criterion == None:
        criterion = rtdetr_criterion()
    if train_dataloader == None:
        train_dataloader = rtdetr_train_dataloader()
    if val_dataloader == None:
        val_dataloader = rtdetr_val_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    if weight_path != None:
        load_tuning_state(weight_path, model)
        
    if use_amp == True:
        scaler = GradScaler()

    if use_ema == True:
        ema_model = ModelEMA(model, decay=0.9999, warmups=2000)
   
    criterion.to(device)  

    lr_scheduler = lr_schedulers.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1) 
    
    print("Start training")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
    best_stat = {'epoch': -1, }

    start_time = time.time()
    last_epoch = 0
    for epoch in range(last_epoch + 1, epoch):
        if dist.is_dist_available_and_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, train_dataloader, optimizer, device, epoch,
            clip_max_norm=0.1, print_freq=100, ema=ema_model, scaler=scaler)

        lr_scheduler.step()
        
        dist.save_on_master(state_dict(epoch, model, optimizer, scaler, ema_model, lr_scheduler), os.path.join(save_dir, f'{epoch}.pth'))

        module = ema_model.module if use_amp == True else model
        test_stats, coco_evaluator = val(model=module, weight_path=None, criterion=criterion, val_dataloader=val_dataloader)

        # TODO 
        for k in test_stats.keys():
            if k in best_stat:
                best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                best_stat[k] = max(best_stat[k], test_stats[k][0])
            else:
                best_stat['epoch'] = epoch
                best_stat[k] = test_stats[k][0]
        print('best_stat: ', best_stat)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if dist.is_main_process():
            with (save_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (save_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                save_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def val(model, weight_path, criterion=None, val_dataloader=None):
    if criterion == None:
        criterion = rtdetr_criterion()
    if val_dataloader == None:
        val_dataloader = rtdetr_val_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    if weight_path != None:
        load_tuning_state(weight_path, model)

    criterion.to(device) 

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")

    header = 'Test:'

    iou_types = postprocessor.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(val_dataloader, 10, header):
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


def rtdetr_r18vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r18vd_6x_coco"

    model = rtdetr.rtdetr_r18vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=output_dir)


def rtdetr_r34vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r34vd_6x_coco"

    model = rtdetr.rtdetr_r34vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params = [{'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'weight_decay': 0., 'lr': 0.00001},
              {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00001}, 
              {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}]

    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=output_dir)


def rtdetr_r50vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r50vd_6x_coco"

    model = rtdetr.rtdetr_r50vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=output_dir)
    
    
def rtdetr_r50vd_m_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r50vd_m_6x_coco"

    model = rtdetr.rtdetr_r50vd_m()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=output_dir)


def rtdetr_r101vd_train():
    weight_path = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"
    output_dir = "./output/rtdetr_r101vd_6x_coco"

    model = rtdetr.rtdetr_r101vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    optimizer=AdamW(params=get_optim_params(params, model), lr=0.00001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=output_dir)

