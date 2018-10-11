#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mtcnn_detector import MtcnnDetector
import os
import cv2
import mxnet as mx


mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
ctx = mx.gpu(0)
det_threshold = [0.6,0.7,0.8]

# detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)
# detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
detector = MtcnnDetector()

img_path = "Tom_Hanks_54745.png"
img_path = "a.jpg"
img = cv2.imread(img_path)
# print img
ret = detector.detect_face(img, det_type=1)
bbox, landmark = ret
# print bbox
pointx = landmark[0][:5]
pointy = landmark[0][5:]
for x, y in zip(pointx, pointy):
    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
