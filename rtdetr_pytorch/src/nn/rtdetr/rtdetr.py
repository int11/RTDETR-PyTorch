"""by lyuwenyu
"""

import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
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

