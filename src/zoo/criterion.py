"""
Copyright (c) 2025 int11. All Rights Reserved.
"""

from src.nn.rtdetr.matcher import HungarianMatcher
from src.nn.rtdetr.rtdetr_criterion import RTDETRCriterion


def rtdetr_criterion():
    matcher = HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                               use_focal_loss=True,
                               alpha=0.25,
                               gamma=2.0)
    
    criterion = RTDETRCriterion(matcher=matcher,
                             weight_dict= {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                             losses= ['vfl', 'boxes'],
                             alpha= 0.75,
                             gamma= 2.0)
    return criterion