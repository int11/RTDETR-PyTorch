"""by lyuwenyu
"""

import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.nn.backbone.presnet import PResNet
from src.core import register

from rtest.utils import *

@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        # 이 코드가 왜여기에있는지 모르겠음. dataloader transform에서 처리해야함...
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        
        vprint(x, 'input image')
        x = self.backbone(x)
        vechook.hook(x, 'backbone output')
        x = self.encoder(x)
        vechook.hook(x, 'encoder output')
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


def rtdetr_r50vd_backbone(
        depth=50, 
        variant='d', 
        freeze_at=0, 
        return_idx=[1, 2, 3], 
        num_stages=4, 
        freeze_norm=True, 
        pretrained=True):
    
    return PResNet(
        depth=depth, 
        variant=variant, 
        freeze_at=freeze_at,
        return_idx=return_idx, 
        num_stages=num_stages, 
        freeze_norm=freeze_norm, 
        pretrained=pretrained)


def rtdetr_r50vd_encoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1,
        act='silu',
        eval_spatial_size=[640, 640]):
    
    return HybridEncoder(
        in_channels=in_channels, 
        feat_strides=feat_strides, 
        hidden_dim=hidden_dim,
        use_encoder_idx=use_encoder_idx, 
        num_encoder_layers=num_encoder_layers, 
        nhead=nhead, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout, 
        enc_act=enc_act, 
        pe_temperature=pe_temperature, 
        expansion=expansion, 
        depth_mult=depth_mult, 
        act=act, 
        eval_spatial_size=eval_spatial_size)


def rtdetr_r50vd_decoder(
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_levels=3,
        num_queries=300,
        num_decoder_layers=6,
        num_denoising=100,
        eval_idx=-1,
        eval_spatial_size=[640, 640]):

    return RTDETRTransformer(
        feat_channels=feat_channels,
        feat_strides=feat_strides,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        num_denoising=num_denoising,
        eval_idx=eval_idx,
        eval_spatial_size=eval_spatial_size)


def rtdetr_r50vd():
    backbone = rtdetr_r50vd_backbone()
    encoder = rtdetr_r50vd_encoder()
    decoder = rtdetr_r50vd_decoder()

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model


def rtdetr_r18vd():
    backbone = rtdetr_r50vd_backbone(depth=18, freeze_at=-1, freeze_norm=False)
    encoder = rtdetr_r50vd_encoder(in_channels=[128, 256, 512], expansion=0.5)
    decoder = rtdetr_r50vd_decoder(num_decoder_layers=3)
    
    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model


def rtdetr_r34vd():
    backbone = rtdetr_r50vd_backbone(depth=34, freeze_at=-1, freeze_norm=False)
    encoder = rtdetr_r50vd_encoder(in_channels=[128, 256, 512], expansion=0.5)
    decoder = rtdetr_r50vd_decoder(num_decoder_layers=4)

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model


def rtdetr_r50vd_m():
    backbone = rtdetr_r50vd_backbone()
    encoder = rtdetr_r50vd_encoder(expansion=0.5)
    decoder = rtdetr_r50vd_decoder(eval_idx=2)

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model


def rtdetr_r101vd():
    backbone = rtdetr_r50vd_backbone(depth=101)
    encoder = rtdetr_r50vd_encoder(hidden_dim=384, dim_feedforward=2048)
    decoder = rtdetr_r50vd_decoder(feat_channels=[384, 384, 384])

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model
