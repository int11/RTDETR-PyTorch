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

__all__ = ['RTDETR', ]


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


def rtdetr_r50vd():
    backbone = PResNet(depth=50,
                      variant='d',
                      freeze_at=0,
                      return_idx=[1, 2, 3],
                      num_stages=4,
                      freeze_norm=True,
                      pretrained=True)
    
    encoder = HybridEncoder(
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
        eval_spatial_size=[640, 640]
    )

    decoder = RTDETRTransformer(
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_levels=3,
        num_queries=300,
        num_decoder_layers=6,
        num_denoising=100,
        eval_idx=-1,
        eval_spatial_size=[640, 640]
    )

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    )

    return model