from src.optim.optim import AdamW
import torch.nn as nn


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


def rtdetr_optimizer(model, params, lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001):
    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    optimizer = AdamW(params=get_optim_params(params, model), lr=lr, betas=betas, weight_decay=weight_decay)
    return optimizer


def rtdetr_r18vd_optimizer(model):
    params= [{'params': '^(?=.*backbone)(?=.*norm).*$', 'lr': 0.00001, 'weight_decay': 0.},
             {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.00001},
             {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$', 'weight_decay': 0.}]
    
    return rtdetr_optimizer(model=model, params=params)

def rtdetr_r34vd_optimizer(model):
    params = [{'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'weight_decay': 0., 'lr': 0.00001},
              {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00001}, 
              {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.}]

    return rtdetr_optimizer(model=model, params=params)


def rtdetr_r50vd_optimizer(model):
    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    return rtdetr_optimizer(model=model, params=params)


def rtdetr_r50vd_m_optimizer(model):
    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    return rtdetr_optimizer(model=model, params=params)


def rtdetr_r101vd_optimizer(model):
    params= [{'params': 'backbone', 'lr': 0.00001},
             {'params': '^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.},
             {'params': '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', 'weight_decay': 0.}]

    return rtdetr_optimizer(model=model, params=params, lr=0.00001)