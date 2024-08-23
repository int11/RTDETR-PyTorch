"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from datetime import datetime
import re
import src.misc.dist as dist
from typing import Dict
import torch.optim.lr_scheduler as lr_scheduler


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


def load_tuning_state(path, model, ema_model=None):
    """only load model for tuning and skip missed/dismatched keys
    """
    state = torch.hub.load_state_dict_from_url(path, map_location='cpu') if 'http' in path else torch.load(path, map_location='cpu')

    infos = dist.de_parallel(model).load_state_dict(state['model'], strict=False)
    print(f'Load model.state_dict, {infos}')

    if 'ema' in state:
        if ema_model is None:
            raise RuntimeError('WARNING, ema model weight exist in file but flag is use_ema=False, skip loading ema model')
        else:
            infos = dist.de_parallel(ema_model).load_state_dict(state['ema'], strict=False)
            print(f'Load ema_model.state_dict, {infos}')

    return state['last_epoch']

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

def state_dict(last_epoch, model, ema_model=None):
    '''current train info state dict 
    '''
    state = {}
    state['model'] = dist.de_parallel(model).state_dict()
    state['date'] = datetime.now().isoformat()
    state['last_epoch'] = last_epoch

    if ema_model is not None:
        state['ema'] = ema_model.state_dict()

    return state
