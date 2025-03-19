"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

import torch
from src.data.coco.coco_dataset import CocoDetection_share_memory
from src.data import transforms as T


def coco_train_dataset(
        img_folder="./dataset/coco/train2017/",
        ann_file="./dataset/coco/annotations/instances_train2017.json",
        range_num=None,
        dataset_class=CocoDetection_share_memory,
        **kwargs):
    
    train_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms = T.Compose([T.RandomPhotometricDistort(p=0.5), 
                                T.RandomZoomOut(fill=0), 
                                T.RandomIoUCrop(p=0.8),
                                T.SanitizeBoundingBoxes(min_size=1),
                                T.RandomHorizontalFlip(),
                                T.Resize(size=[640, 640]),
                                T.SanitizeBoundingBoxes(min_size=1),
                                T.ConvertPILImage(dtype='float32', scale=True),
                                T.ConvertBoxes(fmt='cxcywh', normalize=True)]),
        remap_mscoco_category=True, 
        **kwargs)
    
    if range_num != None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(range_num))

    return train_dataset


def coco_val_dataset(
        img_folder="./dataset/coco/val2017/",
        ann_file="./dataset/coco/annotations/instances_val2017.json",
        range_num=None,
        dataset_class=CocoDetection_share_memory,
        **kwargs):
    
    val_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=T.Compose([T.Resize(size=[640, 640]), 
                              T.ConvertPILImage(dtype='float32', scale=True)]),
        remap_mscoco_category=True,
        **kwargs)
    
    if range_num != None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(range_num))

    return val_dataset