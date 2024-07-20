import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
from src.solver.det_solver import DetSolver
from rtdetr_pytorch.approximation.utils import *
def main(config, resume, tuning, amp, test_only):
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


if __name__ == '__main__':
    Setting.print_shape = True
    #변수 초기화 'Only support from_scrach or resume or tuning at one time'
    config = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'  #설정 파일 경로
    resume = None  # resume = '../checkpoint'
    tuning = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth' # 저장된 가중치 경로
    amp = True # 자동 혼합 정밀도(Automatic Mixed Precision, AMP) FP16 FP32 섞어서 사용. 메모리 사용 감소, 에너지 사용 감소, 계산 속도 향상의 장점
    test_only = True

    main(config, resume, tuning, amp, test_only)
    


    

