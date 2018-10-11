#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import numpy as np
from annoy import AnnoyIndex
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "images")
FACE_IMAGE_PATH = os.path.join(ROOT_DIR, "images", "faces")
from detector_wrapper import Detector
from embedding_wrapper import Embedding
from pose_wrapper import Headpose
from config import Config
from face_preprocess import preprocess
import argparse
import pandas as pd


config = Config()
face_db_pd = [x[:-4] for x in pd.read_csv(config.FACE_NAMES, names=["object_name"]).object_name]
detector = Detector()
embedding = Embedding()
headpose = Headpose()
mtcnn_detector = Detector(detector_model_num=1)


parser = argparse.ArgumentParser(description="program param")
parser.add_argument("--video_path", default=os.path.join(config.IMAGE_PATH, "data", "201/3_15900_16230.mp4"), help="video absolute path")
parser.add_argument("--output_path", default=config.DETECTION_OUTPUT_FILE, help="csv result file absolute path")
args = parser.parse_args()


def main():
    # load face features
    u = AnnoyIndex(512, metric="euclidean")
    u.load(config.FACE_FEATURES)

    video = args.video_path
    f = open(args.output_path, "w")

    fvs = FileVideoStream(video).start()
    time.sleep(1.0)
    fps = FPS().start()

    count = 0
    frame_count = 0
    while fvs.more():
        img = fvs.read()
        f.write("frame_{}".format(frame_count))
        img = cv2.resize(img, None, fx=config.RESIZE_IMAGE, fy=config.RESIZE_IMAGE)
        bboxlist = detector.detect(img)
        for b in bboxlist:
            x1, y1, x2, y2, s = b
            if (s > config.DETECTION_THRESHOLD):
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                width = x2 - x1
                height = y2 - y1

                if width >= config.MIN_SIZE and height >= config.MIN_SIZE:
                    ret = mtcnn_detector.detect_face(img[y1:y2, x1:x2], det_type=1)
                    if ret is None:
                        continue
                    face_image = "frame_" + str(frame_count) + "_face_" + str(count) + ".jpg"
                    count += 1
                    f.write(",object_{},position,{},{},{},{}".format(count,x1,x2,y1,y2))
                    bbox, landmarks = ret
                    if landmarks is None:
                        continue

                    pointx = landmarks[0][:5]
                    pointy = landmarks[0][5:]
                    pointx_img_space = map(lambda x: x + x1, pointx)
                    pointy_img_space = map(lambda y: y + y1, pointy)
                    landmarks_img_space = list(pointx_img_space) + list(pointy_img_space)
                    bbox_process = np.array([x1, y1, x2, y2])
                    landmarks_process = np.array(landmarks_img_space).reshape((2,5)).T
                    nimg = preprocess(img, bbox_process, landmarks_process, image_size="112,112")

                    e = embedding.get_feature(nimg)
                    match = u.get_nns_by_vector(e, config.TOP_ACC_NUM, include_distances=True)

                    pose = headpose.get_pose(nimg)
                    cv2.imwrite(os.path.join(FACE_IMAGE_PATH, "aligned_" + face_image), nimg)
                    f.write(",pose,{},{},{}".format(pose[0], pose[1], pose[2]))

                    identified_face_top_1 = face_db_pd[(match[0][0])]
                    score_top_1 = match[1][0]
                    if score_top_1 < config.MATCH_THRESHOLD:
                        cv2.putText(img, identified_face_top_1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                        for i in range(config.TOP_ACC_NUM):
                            identified_face = face_db_pd[(match[0][i])]

                            if (abs(pose[0]) > config.POSE_THRESHOLD or abs(pose[1]) > config.POSE_THRESHOLD or abs(pose[2]) > config.POSE_THRESHOLD):
                                continue
                            if (match[1][i] < config.MATCH_THRESHOLD):
                                f.write(",{},{}".format(identified_face, match[1][i]))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)

        frame_count += 1
        f.write("\n")
        cv2.imshow("img_window", img)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    cv2.destroyAllWindows()
    fvs.stop()
    f.close()


if __name__ == '__main__':
    main()
