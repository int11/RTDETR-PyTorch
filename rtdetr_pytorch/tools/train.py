import os 
import sys
from torch.cuda.amp import GradScaler, autocast
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.zoo.rtdetr.matcher import HungarianMatcher
import src.misc.dist as dist 
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_solver import DetSolver
from rtest.utils import *
from src.zoo.rtdetr.rtdetr import RTDETR
from typing import Dict

def main():
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
            tuning=tuning
        )

    solver = DetSolver(cfg)

    if test_only:
        solver.val()
    else:
        solver.fit()


def load_tuning_state(path, model):
    def matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    """only load model for tuning and skip missed/dismatched keys
    """
    if 'http' in path:
        state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        state = torch.load(path, map_location='cpu')

    module = dist.de_parallel(model)
    
    # TODO hard code
    if 'ema' in state:
        stat, infos = matched_state(module.state_dict(), state['ema']['module'])
    else:
        stat, infos = matched_state(module.state_dict(), state['model'])

    module.load_state_dict(stat, strict=False)
    print(f'Load model.state_dict, {infos}')


def main1(weight_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    last_epoch = -1

    model = RTDETR.rtdetr_r50vd()
    model.to(device)
    model = dist.warp_model(model, find_unused_parameters=False, sync_bn=True)
        
    load_tuning_state(weight_path)

    matcher = HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                               # use_focal_loss: True 
                               alpha=0.25,
                               gamma=2.0)
    
    criterion = SetCriterion(matcher=matcher,
                             weight_dict= {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,},
                             losses= ['vfl', 'boxes', ],
                             alpha= 0.75,
                             gamma= 2.0)
    
    criterion.to(device)

    postprocessor = RTDETRPostProcessor(num_top_queries= 300)

    scaler = GradScaler()

    

if __name__ == '__main__':
    main()
    Setting.print_shape = True
    #변수 초기화 'Only support from_scrach or resume or tuning at one time'
    config = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'  #설정 파일 경로
    resume = None  # resume = '../checkpoint'
    path = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth' # 저장된 가중치 경로
    amp = True # 자동 혼합 정밀도(Automatic Mixed Precision, AMP) FP16 FP32 섞어서 사용. 메모리 사용 감소, 에너지 사용 감소, 계산 속도 향상의 장점
    test_only = False


    main1(path=path)