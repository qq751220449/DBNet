# -*- coding: utf-8 -*-
# @Time    : 2020/12/02 11:14
# @Author  : liyujun

from .DBHead import DBHead


__all__ = ['build_head']

support_head = {
    "DBHead": DBHead
}


def build_head(head_type, **kwargs):
    assert head_type in support_head.keys(), f'all support head is {support_head.keys()}'
    head = support_head[head_type](**kwargs)
    return head
