import torch 
from src.data.dataloader import DataLoader
from src.core import YAMLConfig, yaml_utils
from src.solver.det_solver import DetSolver
from rtest.utils import *
from src.zoo.rtdetr.rtdetr import RTDETR
import numpy as np 


import src.misc.dist as dist 
import torch.nn as nn

import torch.nn.functional as F 

def encoder_forward(self, feats):
    assert len(feats) == len(self.in_channels)
    proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
    
    # encoder
    if self.num_encoder_layers > 0:
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

            memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
            # print([x.is_contiguous() for x in proj_feats ])

    # broadcasting and fusion
    inner_outs = [proj_feats[-1]]
    for idx in range(len(self.in_channels) - 1, 0, -1):
        feat_high = inner_outs[0]
        feat_low = proj_feats[idx - 1]
        feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
        inner_outs[0] = feat_high

        # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
        upsample_feat = feat_high

        inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
        inner_outs.insert(0, inner_out)

    outs = [inner_outs[0]]
    for idx in range(len(self.in_channels) - 1):
        feat_low = outs[-1]
        feat_high = inner_outs[idx + 1]

        # feat_low = F.interpolate(feat_low, scale_factor=2., mode='nearest')
        downsample_feat = self.downsample_convs[idx](feat_low)
        # feat_low = F.interpolate(feat_low, scale_factor=2., mode='nearest')
        

        out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
        outs.append(out)

    return outs


def model_forward(self, x, targets=None):
    if self.multi_scale and self.training:
        sz = np.random.choice(self.multi_scale)
        x = F.interpolate(x, size=[sz, sz])

    x = self.backbone(x)
    x[0] = F.interpolate(x[0], (20,20))
    x[1] = F.interpolate(x[1], (20,20))
    x = self.encoder(x)
    x = self.decoder(x, targets)

    return x

class test2Solver(DetSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def change_forward(self, layer, forward_func):
        layer.forward = lambda *args, self=layer, **kwargs: forward_func(self, *args, **kwargs)

    @torch.no_grad()
    def val1(self):
        print(self.cfg.yaml_cfg['model'])
        self.setup()
        dataloader, device = self.cfg.train_dataloader, self.device
        # shuffle을 False로 설정하여 새 DataLoader 객체 생성
        dataloader = DataLoader(dataloader.dataset, batch_size=4, shuffle=False, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory, collate_fn=dataloader.collate_fn)

        model = self.model
        self.change_model_forward(model)
        model.eval()

        
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            original_index = [int(i['image_id']) for i in targets]
            outputs = model(samples)

    @torch.no_grad()
    def val(self, ):
        from src.data import get_coco_api_from_dataset
        from src.solver.det_engine import evaluate

        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        
        self.change_forward(module.encoder, encoder_forward)
        self.change_forward(module, model_forward)

        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        

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
    
    test2Solver(cfg).val()