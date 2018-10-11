#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from thirdparty.detector import SFDDetector
from thirdparty.detector import MtcnnDetector


DETECTOR_MODELS = [SFDDetector, MtcnnDetector]


class Detector(object):
    """
    Detector Wrapper
    """
    def __init__(self, detector_model_num=0):
        self.detector_model_num = detector_model_num
        self.detector_class = DETECTOR_MODELS[self.detector_model_num]()


    def __getattr__(self, attr):
        return self.detector_class.__getattribute__(attr)
