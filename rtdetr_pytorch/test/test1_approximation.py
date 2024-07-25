import os 
import sys

import torch 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data.dataloader import DataLoader
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver.det_solver import DetSolver
import torch.nn as nn
from rtdetr_pytorch.test.utils import *

class CustomModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModule, self).__init__()
        # input_size와 output_size는 flat한 텐서의 차원을 기반으로 합니다.
        self.linear = [self.mklayer(input_size[i], output_size[i]) for i in range(3)]


    def mklayer(self, input_size, output_size):
        layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
        return layer
    
    def forward(self, x1, x2, x3):
        # 각 텐서에 대해 nn.Linear 레이어 적용
        x1_out = self.linear[0](x1)
        x2_out = self.linear[1](x2)
        x3_out = self.linear[2](x3)

        return x1_out, x2_out, x3_out
    
class test1Solver(DetSolver):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def get_hidden_vec(self):
        self.setup()
        dataloader, device = self.cfg.train_dataloader, self.device
        # shuffle을 False로 설정하여 새 DataLoader 객체 생성
        dataloader = DataLoader(dataloader.dataset, batch_size=4, shuffle=False, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory, collate_fn=dataloader.collate_fn)

        model = self.ema.module if self.ema else self.model
        model.eval()

        
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            original_index = [int(i['image_id']) for i in targets]
            print(original_index)
            with vechook():
                outputs = model(samples)
            
            yield vechook.variable
            
    def train(self):
        model = None
        for i in self.get_hidden_vec():
            x, t = i['backbone output'], i['encoder output']

            x = [i.view(i.size(0), -1) for i in x]
            t = [i.view(i.size(0), -1) for i in t]
            
            input_shape = [i.shape[-1] for i in x]
            output_shape = [i.shape[-1] for i in t]
            if model == None:
                model = CustomModule(input_shape, output_shape)
            y = model(*x)
            print(y)



if __name__ == '__main__':
    Setting.print_shape = True
    #분산 프로세스 초기화
    dist.init_distributed()

    #변수 초기화 'Only support from_scrach or resume or tuning at one time'
    config = 'rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml'  #설정 파일 경로
    resume = None  # resume = '../checkpoint'
    tuning = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth' # 저장된 가중치 경로
    amp = True # 자동 혼합 정밀도(Automatic Mixed Precision, AMP) FP16 FP32 섞어서 사용. 메모리 사용 감소, 에너지 사용 감소, 계산 속도 향상의 장점
    test_only = False

    #첫번재 인자로 받은 설정파일에 이후의 인자들을 merge 하여 설정파일 생성
    cfg = YAMLConfig(
            config,
            resume=resume, 
            use_amp=amp,
            tuning=tuning
        )
    
    test1Solver(cfg).train()
