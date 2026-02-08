"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import argparse
from utils import str2bool


def get_args_parser():
    """
    Create argument parser for training and validation
    """
    parser = argparse.ArgumentParser(description='RT-DETR Training and Validation')
    
    parser.add_argument('--weight_path', '-w', type=str, default=None,
                        help='path to the weight file (default: None)')

    parser.add_argument('--save_dir', '-s', type=str, default='output/rtdetr_r18vd_6x_coco',
                        help='path to the weight save directory (default: output/rtdetr_r18vd_6x_coco)')

    parser.add_argument('--dataset_dir', type=str, default='datasets/coco',
                        help='path to the dataset directory (default: dataset/coco). '
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

    parser.add_argument('--epoch', type=int, default=72,
                        help='When test-only is False, this is the number of epochs to train (default: 72)')

    parser.add_argument('--model_type', type=str, default='r18vd',
                        choices=['r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd'],
                        help='choose the model type (default: r18vd)')

    parser.add_argument('--use_focal_loss', type=str2bool, default=True,
                        help='enable focal loss (default: True)')

    parser.add_argument('--remap_mscoco_category', type=str2bool, default=True,
                        help='if True, remap MSCOCO category ids (default: False)')
    
    return parser
