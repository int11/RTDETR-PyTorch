"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

from ..rtdetr import RTDETRTransformer
from collections import OrderedDict
from torch import nn

class RTDETRTransformerv2(RTDETRTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(args.hidden_dim, args.hidden_dim)),
            ('norm', nn.LayerNorm(args.hidden_dim,)),
        ]))