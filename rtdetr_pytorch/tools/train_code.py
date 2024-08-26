import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.zoo import model as rtdetr_zoo
from src.zoo import val, fit, rtdetr_val_dataset, rtdetr_train_dataset, rtdetr_train_dataloader, rtdetr_val_dataloader, rtdetr_r50vd_optimizer

from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist

from rtest.utils import *


def main(test_only=False):
    dist.init_distributed()
    
    weight_path = "output/rtdetr_r18vd_6x_coco/78.pth"
    save_dir = "output/rtdetr_r18vd_6x_coco"
    batch_size = 32
    num_workers = 4


    model = rtdetr_zoo.rtdetr_r18vd()
    
    #TODO CocoDetection_share_memory eval bug
    val_dataset = rtdetr_val_dataset(dataset_class=CocoDetection)
    val_dataloader = rtdetr_val_dataloader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)

    if test_only:
        val(model, weight_path, val_dataloader=val_dataloader)
    else:
        optimizer = rtdetr_r50vd_optimizer(model=model)
        train_dataloader = rtdetr_train_dataloader(batch_size=batch_size, num_workers=num_workers)
        fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=save_dir, train_dataloader=train_dataloader, val_dataloader=val_dataloader, use_amp=True, use_ema=True, epoch=100)


if __name__ == '__main__':
    main(False)