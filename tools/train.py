"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import zoo
from utils import fit, val, str2bool
from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist_utils
from src.data.dataloader import DataLoader, BatchImageCollateFuncion
import argparse


def main():
    args = parser.parse_args()


    dist_utils.init_distributed()
    
    model = getattr(zoo.model, args.model_type)()
    optimizer = getattr(zoo.optimizer, args.model_type)(model)

    #TODO There is a slow on a dataset that is not a CocoDetection class, need to fix this
    val_dataset = zoo.coco_val_dataset(
        img_folder=os.path.join(args.dataset_dir, "val2017"),
        ann_file=os.path.join(args.dataset_dir, "annotations/instances_val2017.json"), 
        dataset_class=CocoDetection)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, 
                                collate_fn=BatchImageCollateFuncion())

    if args.val:
        val(model=model, 
            weight_path=args.weight_path, 
            val_dataloader=val_dataloader,
            use_amp=args.amp,
            use_ema=args.ema)
    else:
        train_dataset = zoo.coco_train_dataset(
            img_folder=os.path.join(args.dataset_dir, "train2017"),
            ann_file=os.path.join(args.dataset_dir, "annotations/instances_train2017.json"), range_num=500)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True, 
                                      collate_fn=BatchImageCollateFuncion(scales=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800], stop_epoch=71))
        fit(
            model=model, 
            weight_path=args.weight_path,
            optimizer=optimizer, 
            save_dir=args.save_dir, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            use_amp=args.amp, 
            use_ema=args.ema, 
            epoch=args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', '-w', type=str, default=None,
                        help='path to the weight file (default: None)')

    parser.add_argument('--save_dir', '-s', type=str, default='output/rtdetr_r18vd_6x_coco',
                        help='path to the weight save directory (default: output/rtdetr_r18vd_6x_coco)')

    parser.add_argument('--dataset_dir', type=str, default='dataset/coco',
                        help='path to the dataset directory (default: dataset/coco)'
                        'This is the directory that must contains the train2017, val2017, annotations folder')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers (default: 0)')

    parser.add_argument('--val', type=str2bool, default=False,
                        help='if True, only evaluate the model (default: False)')

    parser.add_argument('--amp', type=str2bool, default=True,
                        help='When GPU is available, use Automatic Mixed Precision (default: True)')
    
    parser.add_argument('--ema', type=str2bool, default=True,
                        help='Use Exponential Moving Average (default: True)')

    parser.add_argument('--epoch', type=int, default=100,
                        help='When test-only is False, this is the number of epochs to train (default: 100)')

    parser.add_argument('--model_type', type=str, default='r18vd',
                        choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd'],
                        help='choose the model type (default: r18vd)')

    main()