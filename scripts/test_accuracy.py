#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys
import cv2
import pandas as pd
from annoy import AnnoyIndex
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "images")
from embedding_wrapper import Embedding
from config import Config


config = Config()
face_db_pd = [x[:-4] for x in pd.read_csv(config.FACE_NAMES, names=["object_name"]).object_name]


def main():
    u = AnnoyIndex(512, metric="euclidean")
    u.load(config.FACE_FEATURES)
    embedding = Embedding()

    total = 0
    correct = 0
    for face in face_db_pd:
        current_test_path = os.path.join(config.TEST_PATH, face)
        if (not os.path.exists(current_test_path)):
            continue
        images = [image for image in os.listdir(current_test_path) if image.endswith(".jpg")]
        imgs = [cv2.imread(os.path.join(current_test_path, image)) for image in images]

        total += len(imgs)
        for img in imgs:
            e = embedding.get_feature(img)
            match = u.get_nns_by_vector(e, 1, include_distances=True)
            identified_face = face_db_pd[(match[0][0])]
            if (identified_face == face):
                correct += 1

    print("total test images: {}".format(total))
    print("total correct images: {}".format(correct))
    if (total == 0):
        print("No test images")
    else:
        print("Accuracy: {}%".format(float(correct) * 100 /total))


if __name__ == '__main__':
    main()
