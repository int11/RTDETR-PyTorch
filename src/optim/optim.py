"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']


SGD = optim.SGD
Adam = optim.Adam
AdamW = optim.AdamW


MultiStepLR = lr_scheduler.MultiStepLR
CosineAnnealingLR = lr_scheduler.CosineAnnealingLR
OneCycleLR = lr_scheduler.OneCycleLR
LambdaLR = lr_scheduler.LambdaLR
