import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.zoo import model as rtdetr_zoo
from src.zoo import val, fit, rtdetr_val_dataset, rtdetr_train_dataset, rtdetr_train_dataloader, rtdetr_val_dataloader, rtdetr_r50vd_optimizer

from src.data.coco.coco_dataset import CocoDetection
from src.misc import dist

from rtest.utils import *



def validate():
    weight_path = "output/rtdetr_r18vd_6x_coco/47.pth"
    model = rtdetr_zoo.rtdetr_r18vd()

    val_dataset = rtdetr_val_dataset(dataset_class=CocoDetection)
    val_dataloader = rtdetr_val_dataloader(dataset=val_dataset, batch_size=16, num_workers=0)

    val(model, weight_path, val_dataloader=val_dataloader)


def main():
    dist.init_distributed()
    
    weight_path = None
    save_dir = "output/rtdetr_r18vd_6x_coco"
    
    model = rtdetr_zoo.rtdetr_r18vd()
    optimizer = rtdetr_r50vd_optimizer(model=model)
    

    train_dataloader = rtdetr_train_dataloader(batch_size=8, num_workers=0)
    val_dataloader = rtdetr_val_dataloader(batch_size=4, range_num=1000, num_workers=0)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=save_dir, train_dataloader=train_dataloader, val_dataloader=val_dataloader, use_amp=True, use_ema=True)


if __name__ == '__main__':
    validate()

    