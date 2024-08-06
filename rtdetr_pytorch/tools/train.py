from src.zoo.rtdetr.train import fit, rtdetr_train_dataloader, rtdetr_val_dataloader
from src.zoo.rtdetr.utils import get_optim_params
from src.misc import dist
from src.zoo.rtdetr import *
from src.optim.optim import AdamW
from rtest.utils import *

from src.core import YAMLConfig
from src.solver.det_solver import DetSolver
from src.zoo.rtdetr import rtdetr
from src.optim.optim import AdamW
from src.zoo.rtdetr import rtdetr
from rtest.utils import *

import src.misc.dist as dist

# original implement
def main1():
    Setting.print_shape = True
    #변수 초기화 'Only support from_scrach or resume or tuning at one time'
    config = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'  #설정 파일 경로
    resume = None  # resume = '../checkpoint'
    tuning = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth' # 저장된 가중치 경로
    amp = True # 자동 혼합 정밀도(Automatic Mixed Precision, AMP) FP16 FP32 섞어서 사용. 메모리 사용 감소, 에너지 사용 감소, 계산 속도 향상의 장점
    test_only = False

    #분산 프로세스 초기화
    dist.init_distributed()

    #첫번재 인자로 받은 설정파일에 이후의 인자들을 merge 하여 설정파일 생성
    cfg = YAMLConfig(
            config,
            resume=resume, 
            use_amp=amp,
            tuning=tuning)

    solver = DetSolver(cfg)

    if test_only:
        solver.val()
    else:
        solver.fit()

# custom implement
def main2():
    weight_path = None
    save_dir = "./output/rtdetr_r18vd_6x_coco"

    model = rtdetr.rtdetr_r18vd()
    model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)

    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    optimizer = AdamW(params=get_optim_params(params, model), lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)

    fit(model=model, weight_path=weight_path, optimizer=optimizer, save_dir=save_dir, val_dataloader=rtdetr_val_dataloader(range_num=1000))


if __name__ == '__main__':
    main2()

    