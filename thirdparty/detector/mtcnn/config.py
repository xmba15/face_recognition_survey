#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os


_DIRECTORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "mtcnn-model")


class Config(object):
    def __init__(self):
        self.MODEL_PATH = _MODEL_PATH


    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
