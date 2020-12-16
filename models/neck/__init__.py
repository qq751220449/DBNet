# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 11:14
# @Author  : liyujun

from .FPN import FPN
from .SPP import SPP


__all__ = ['build_neck']

support_neck = {
    "FPN": FPN,
    "SPP": SPP,
}


def build_neck(neck_type, **kwargs):
    assert neck_type in support_neck.keys(), f'all support neck is {support_neck.keys()}'
    neck = support_neck[neck_type](**kwargs)
    return neck
