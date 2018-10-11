#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys
import cv2
from annoy import AnnoyIndex
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
from config import Config
from embedding_wrapper import Embedding


config = Config()


def human_sort(s):
    """Sort list the way humans do
    """
    import re
    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(pattern, s)]


def main():
    embed = Embedding()
    images = [image for image in
              os.listdir(os.path.join(config.IMAGE_PATH, "face_db"))
              if image.endswith(".png")]
    images.sort(key=human_sort)

    with open(config.FACE_NAMES, 'w') as f:
        [f.write(image + '\n') for image in images]

    imgs = [cv2.imread(os.path.join(config.IMAGE_PATH, "face_db", image))
            for image in images]

    t = AnnoyIndex(512, metric="euclidean")

    for i, img in enumerate(imgs):
        t.add_item(i, embed.get_feature(img))
    t.build(10)

    # rewrite face_db.ann
    t.save(config.FACE_FEATURES)


if __name__ == '__main__':
    main()
