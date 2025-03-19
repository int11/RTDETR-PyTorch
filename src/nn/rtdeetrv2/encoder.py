"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

from ..rtdetr import HybridEncoder
import torch.nn as nn
from collections import OrderedDict


class HybridEncoderv2(HybridEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in args.in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, args.hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(args.hidden_dim))
            ]))
            self.input_proj.append(proj)