"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

from src.optim.optim import AdamW
import torch.nn as nn
import re

def get_optim_params(params, model: nn.Module):
    '''
    E.g.:
        ^(?=.*a)(?=.*b).*$         means including a and b
        ^((?!b.)*a((?!b).)*$       means including a but not b
        ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
    '''

    if params == None:
        return model.parameters() 

    assert isinstance(params, list), ''

    param_groups = []
    visited = []
    for pg in params:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''

    return param_groups


def r18vd(model, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml
    params = [
        {'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'weight_decay': 0., 'lr': 0.00001},
        {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00001},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}
    ]
    return AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)


def r34ad(model, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r34vd_6x_coco.yml
    params = [
        {'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'weight_decay': 0., 'lr': 0.00001},
        {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00001},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}
    ]
    return AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)


def r50vd(model, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml
    params = [
        {'params': '^(?=.*backbone)(?!.*(?:norm|bn)).*$', 'lr': 0.00001},
        {'params': '^(?=.*backbone)(?=.*(?:norm|bn)).*$', 'weight_decay': 0., 'lr': 0.00001},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}
    ]
    return AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)


def r50vd_m(model, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml
    params = [
        {'params': '^(?=.*backbone)(?!.*norm|bn).*$','lr': 0.00001},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}
    ]
    return AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)


def r101vd(model, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml
    params = [
        {'params': '^(?=.*backbone)(?!.*norm|bn).*$','lr': 0.000001},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$', 'weight_decay': 0.}
    ]
    return AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)
