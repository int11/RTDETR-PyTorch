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
                                T.SanitizeBoundingBox(min_size=1),
                                T.RandomHorizontalFlip(),
                                T.Resize(size=[640, 640]),
                                # transforms.Resize(size=639, max_size=640),
                                # # transforms.PadToSize(spatial_size=640),
                                T.ToImageTensor(),
                                T.ConvertDtype(),
                                T.SanitizeBoundingBox(min_size=1),
                                T.ConvertBox(out_fmt='cxcywh', normalize=True)]),
        return_masks=False,
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
                                T.ToImageTensor(), 
                                T.ConvertDtype()]),
        return_masks=False,
        remap_mscoco_category=True,
        **kwargs)
    
    if range_num != None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(range_num))

    return val_dataset