# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 11:14
# @Author  : liyujun

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, deformable_resnet18, deformable_resnet50
from .atros_resnet import atros_resnet18, atros_resnet101


__all__ = ['build_backbone']

support_backbone = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "deformable_resnet18": deformable_resnet18,
    "deformable_resnet50": deformable_resnet50,
    "atros_resnet18": atros_resnet18,
    "atros_resnet101": atros_resnet101,
}


def build_backbone(backbone_type, **kwargs):
    assert backbone_type in support_backbone.keys(), f'all support backbone is {support_backbone.keys()}'
    backbone = support_backbone[backbone_type](**kwargs)
    return backbone

