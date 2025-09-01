"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import os
import sys

import torch
from torch.cuda.amp import GradScaler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.nn.rtdetr.criterion import RTDETRCriterion
from src.nn.rtdetr.matcher import HungarianMatcher
from src.nn.rtdetr.postprocessor import RTDETRPostProcessor

from src import zoo
from utils import Tee, fit, val, load_tuning_state
from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist_utils
from src.data.dataloader import DataLoader, BatchImageCollateFuncion
from options import get_args_parser
import torch.optim.lr_scheduler as lr_schedulers
from src.optim.ema import ModelEMA


def main(args):
    # model
    model = getattr(zoo.model, args.model_type)()

    # optimizer
    optimizer = getattr(zoo.optimizer, args.model_type)(model)

    # loss function
    matcher = HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                               use_focal_loss=args.use_focal_loss,
                               alpha=0.25,
                               gamma=2.0)
    criterion = RTDETRCriterion(matcher=matcher,
                             weight_dict= {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                             losses= ['vfl', 'boxes'],
                             alpha= 0.75,
                             gamma= 2.0)
    
    # postprocessor
    postprocessor = RTDETRPostProcessor(
        num_classes=80,
        use_focal_loss=args.use_focal_loss,
        num_top_queries=300,
        remap_mscoco_category=args.remap_mscoco_category
    )
    
    # amp
    scaler = GradScaler() if args.amp == True else None

    # ema
    ema_model = ModelEMA(model, decay=0.9999, warmups=2000) if args.ema == True else None




    #TODO There is a slow on a dataset that is not a CocoDetection class, need to fix this
    val_dataset = zoo.coco_val_dataset(
        img_folder=os.path.join(args.dataset_dir, "val2017"),
        ann_file=os.path.join(args.dataset_dir, "annotations/instances_val2017.json"), 
        dataset_class=CocoDetection)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, 
                                collate_fn=BatchImageCollateFuncion())

    # evaluation
    if args.val: 
        if args.weight_path != None:
            if 'http' in args.weight_path:
                state = torch.hub.load_state_dict_from_url(args.weight_path, map_location='cpu')
            else:
                state = torch.load(args.weight_path, map_location='cpu')

            if args.ema == True:
                model.load_state_dict(state['ema']['module'], strict=False)
            else:
                model.load_state_dict(state['model'], strict=False)

        val(model=model,
            criterion=criterion,
            val_dataloader=val_dataloader,
            postprocessor=postprocessor,
            scaler=scaler)
    # train
    else:
        train_dataset = zoo.coco_train_dataset(
            img_folder=os.path.join(args.dataset_dir, "train2017"),
            ann_file=os.path.join(args.dataset_dir, "annotations/instances_train2017.json"))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, 
                                      collate_fn=BatchImageCollateFuncion(scales=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800], stop_epoch=71))
        lr_scheduler = lr_schedulers.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1) 
        
        if args.weight_path != None:
            last_epoch = load_tuning_state(args.weight_path, model, ema_model, optimizer, lr_scheduler, scaler)

        fit(model=model, 
            criterion=criterion,
            optimizer=optimizer, 
            save_dir=args.save_dir, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            postprocessor=postprocessor,
            ema_model=ema_model,
            scaler=scaler,
            train_epoch=args.epoch,
            resume_epoch=last_epoch)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    dist_utils.init_distributed()

    with Tee(os.path.join(args.save_dir, f'log.txt')):
        print(args)
        main(args)