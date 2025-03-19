"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.nn as nn 

class RTDETR(nn.Module):
    def __init__(self, backbone: nn.Module, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

