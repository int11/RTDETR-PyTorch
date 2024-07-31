import math
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

def attentionWeight_twice_matmul(weight, feature):
    N, Q, K = weight.shape
    QH = QW = int(math.sqrt(Q))
    KH = KW = int(math.sqrt(K))

    # 4 dimension interpolate 
    scale_factor = 2
    attentionWeight_twice = weight.reshape(N, QH, 1, QW, 1, KH, 1, KW, 1).repeat(1,1,scale_factor,1,scale_factor,1,scale_factor,1,scale_factor).reshape(N, Q * 2 ** 2, K * 2 ** 2)

    N, C, H, W = feature.shape
    result = torch.einsum('nij, ncj->nci', attentionWeight_twice, feature.flatten(2))
    result = result.reshape(N, C, H, W)
    return result + feature, attentionWeight_twice


def TransformerEncoderLayer_forward(self, solver, src, src_mask=None, pos_embed=None) -> torch.Tensor:
    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    q = k = self.with_pos_embed(src, pos_embed)


    src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
    solver.attention_weight = _

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    return src

def encoder_forward(self, solver, feats):
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


    proj_feats[1], attentionWeight_twice = attentionWeight_twice_matmul(solver.attention_weight, proj_feats[1])
    proj_feats[0], _ = attentionWeight_twice_matmul(attentionWeight_twice, proj_feats[0])

    # broadcasting and fusion
    inner_outs = [proj_feats[-1]]
    for idx in range(len(self.in_channels) - 1, 0, -1):
        feat_high = inner_outs[0]
        feat_low = proj_feats[idx - 1]
        feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
        inner_outs[0] = feat_high
        upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
        inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
        inner_outs.insert(0, inner_out)

    outs = [inner_outs[0]]
    for idx in range(len(self.in_channels) - 1):
        feat_low = outs[-1]
        feat_high = inner_outs[idx + 1]
        downsample_feat = self.downsample_convs[idx](feat_low)
        out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
        outs.append(out)

    return outs

class test3Solver(DetSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.attention_weight = None

    def change_forward(self, layer, forward_func):
        layer.forward = lambda *args, self=layer, solver=self, **kwargs: forward_func(self, solver, *args, **kwargs)

    def eval(self, ):
        super().eval()
        model = self.ema.module if self.ema else self.model
        self.change_forward(model.encoder.encoder[0].layers[0], TransformerEncoderLayer_forward)
        self.change_forward(model.encoder, encoder_forward)

        test_dataset = torch.utils.data.Subset(self.val_dataloader.dataset, range(1000))
        self.val_dataloader = DataLoader(test_dataset,
                                         batch_size=self.val_dataloader.batch_size,
                                         shuffle=False,
                                         num_workers=self.val_dataloader.num_workers,
                                         pin_memory=self.val_dataloader.pin_memory,
                                         collate_fn=self.val_dataloader.collate_fn)
    
        

if __name__ == '__main__':
    Setting.print_shape = False
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
    
    test3Solver(cfg).val()