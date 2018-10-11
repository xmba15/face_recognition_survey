#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from thirdparty.pose import HopenetPose


HEADPOSE_MODELS = [HopenetPose]


class Headpose(object):
    def __init__(self, headpose_model_num=0):
        self.headpose_model_num = headpose_model_num
        self.headpose_class = HEADPOSE_MODELS[self.headpose_model_num]()

    def __getattr__(self, attr):
        return self.headpose_class.__getattribute__(attr)
