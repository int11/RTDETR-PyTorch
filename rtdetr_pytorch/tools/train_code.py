import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.zoo.train import fit, rtdetr_train_dataloader, rtdetr_val_dataloader
from src.nn.rtdetr.utils import get_optim_params
from src.misc import dist

from rtest.utils import *

from src.core import YAMLConfig
from src.solver.det_solver import DetSolver
from src.optim.optim import AdamW
from src.zoo import model as rtdetr_zoo
from rtest.utils import *

import torch.utils.data as data


def main():
    dist.init_distributed()
    
    weight_path = None
    save_dir = "output/rtdetr_r18vd_6x_coco"
    
    model = rtdetr_zoo.rtdetr_r18vd()

    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    train_dataloader = rtdetr_train_dataloader(batch_size=8, num_workers=0)
    val_dataloader = rtdetr_val_dataloader(batch_size=4, range_num=1000, num_workers=0)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=save_dir, train_dataloader=train_dataloader, val_dataloader=val_dataloader, use_amp=True, use_ema=True)


if __name__ == '__main__':
    main()

    