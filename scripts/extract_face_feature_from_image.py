#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
from config import Config
from detector_wrapper import Detector
from embedding_wrapper import Embedding
from face_preprocess import preprocess
from annoy import AnnoyIndex
import cv2
import argparse
import numpy as np
import pandas as pd
import subprocess


config = Config()
parser = argparse.ArgumentParser(description="extract face feature from image program param")
parser.add_argument("--image_path", default=os.path.join(config.IMAGE_PATH, "ny.jpg"), help="face image absolute path")
parser.add_argument("--face_name", default="unknown", help="name of face image file to be stored in face_db.csv")
args = parser.parse_args()


detector = Detector()
embedding = Embedding()
mtcnn_detector = Detector(detector_model_num=1)
u = AnnoyIndex(512, metric="euclidean")
u.load(config.FACE_FEATURES)
face_db_pd = [x[:-4] for x in pd.read_csv(config.FACE_NAMES, names=["object_name"]).object_name]


def count_face(bbox_list):
    """Count number of bbox in the bbox_list, whose detection score is over the threshold.
       This function is only used internally inside this script.

    Args:
        bbox_list (list): a list of bbox of faces.

    Returns:
        count (int): number of boxes
        bbox (list): the last bounding box in the bbox_list that fits the above criteria
    """
    count = 0
    bbox = None
    for b in bbox_list:
        x1, y1, x2, y2, s = b
        if (s > config.DETECTION_THRESHOLD):
            bbox = b
            count += 1
    return count, bbox


def store_face_feature(image_path, face_name):
    """Detect and store feature vector of the new face into face_db.ann.

    Args:
        image_path (str): absolute path to the face image file
        face_name (str): name of the face to be restored

    Returns:
         (void):
    """
    face_img = cv2.imread(image_path)
    bbox_list = detector.detect(face_img)

    if (face_name in face_db_pd):
        print('\n')
        print("Face name already stored in face name csv. Choose another name!\n")
        exit(0)

    counter, bbox = count_face(bbox_list)

    if (counter != 1):
        print('\n')
        print("Choose image with only one face\n")
        exit(0)

    x1, y1, x2, y2, s = bbox

    if (s <= config.DETECTION_THRESHOLD):
        print('\n')
        print("Choose image with a clearer face\n")
        exit(0)

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    width = x2 - x1
    height = y2 - y1

    if (width < config.MIN_SIZE or height < config.MIN_SIZE):
        print('\n')
        print("Choose image with a bigger face\n")
        exit(0)

    ret = mtcnn_detector.detect_face(face_img[y1:y2, x1:x2], det_type=1)
    bbox, landmarks = ret

    if (landmarks is None):
        print('\n')
        print("Cannot detect the landmarks\n")
        exit(0)

    pointx = landmarks[0][:5]
    pointy = landmarks[0][5:]
    pointx_img_space = map(lambda x: x + x1, pointx)
    pointy_img_space = map(lambda y: y + y1, pointy)
    landmarks_img_space = list(pointx_img_space) + list(pointy_img_space)
    bbox_process = np.array([x1, y1, x2, y2])
    landmarks_process = np.array(landmarks_img_space).reshape((2,5)).T
    nimg = preprocess(face_img, bbox_process, landmarks_process, image_size="{},{}".format(config.FACE_SIZE,config.FACE_SIZE))
    e = embedding.get_feature(nimg)
    match = u.get_nns_by_vector(e, 1, include_distances=True)

    answer = None
    if (match[1][0] < config.MATCH_THRESHOLD):
        print("The person might be in the list already")
        print("Do you really want to add this into the face list. Answer yes or no")
        if sys.version_info[0] < 3:
            answer = raw_input()
        else:
            answer = input()
        while (answer.strip() not in ['yes', 'no']):
            print("Answer yes or no")
            if sys.version_info[0] < 3:
                answer = raw_input()
            else:
                answer = input()

    if answer is None or answer.strip() == 'yes':
        with open(config.FACE_NAMES, 'a') as f:
            f.write(face_name + ".png")
        result_path = os.path.join(config.IMAGE_PATH, "face_db", face_name + ".png")
        cv2.imwrite(result_path, nimg)

        # rewrite face_db.ann
        another_process = "python {}/scripts/extract_face_feature.py".format(ROOT_DIR)
        subprocess.call(another_process, shell=True)


def main(args):
    store_face_feature(args.image_path, args.face_name)


if __name__ == '__main__':
    main(args)
