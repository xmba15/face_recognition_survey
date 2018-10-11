#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os


_DIRECTORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
_IMAGE_PATH = os.path.join(_DIRECTORY_ROOT, "images")
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "models")
_ALIGN_MODEL_PATH = os.path.join(_MODEL_PATH, "landmarks.dat")
_TEST_PATH = os.path.join(_IMAGE_PATH, "test")
_FACE_FEATURES = os.path.join(_IMAGE_PATH, "face_db.ann")
_FACE_NAMES = os.path.join(_IMAGE_PATH, "face_db.csv")
_DETECTION_OUTPUT_FILE = os.path.join(_DIRECTORY_ROOT, "output", "result.csv")


class Config(object):
    def __init__(self):
        # image path
        self.IMAGE_PATH = _IMAGE_PATH

        # align model path (not using now)
        self.ALIGN_MODEL_PATH = _ALIGN_MODEL_PATH

        # test image path
        self.TEST_PATH = _TEST_PATH

        # file that holds features of the faces
        self.FACE_FEATURES = _FACE_FEATURES

        # file that holds name of the face in db
        self.FACE_NAMES = _FACE_NAMES

        # detection accuracy threshold
        self.DETECTION_THRESHOLD = 0.85

        # pose degree threshold
        self.POSE_THRESHOLD = 35

        # min size of the face image to consider
        self.MIN_SIZE = 40

        # threshold of euclidean distance for matching to feature vectors
        self.MATCH_THRESHOLD = 1.2

        self.TOP_ACC_NUM = 2

        # directory to store result
        self.DETECTION_OUTPUT_FILE = _DETECTION_OUTPUT_FILE

        # face size to store at the face db
        self.FACE_SIZE = 112

        # ratio to resize input image
        self.RESIZE_IMAGE = 0.8

    def display(self):
        """
        Display Configuration values.

        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
