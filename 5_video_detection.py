# -*- coding: utf-8 -*-
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files

import cv2
import os

IMG_SRC_DIR = "..//project_video"
IMG_DST_DIR = "..//project_video//debug"

# if turn on this option, you can see the detected boxes from both img frawework & video framework.
# if turn off this option, you can see the detected boxes from video framework only.
DRAW_IMG_DETECTION_RESULT = False

START = 250
if __name__ == "__main__":
    img_files = list_files(IMG_SRC_DIR, pattern="*.jpg", random_order=False, recursive_option=False)    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    
    count = START
    for fname in img_files[START:]:
        img = cv2.imread(fname)
        img_draw = d.run(img, DRAW_IMG_DETECTION_RESULT)

        count_str = "{}".format(count).zfill(5)
        filename = os.path.join(IMG_DST_DIR, "{}.jpg".format(count_str))
        cv2.imwrite(filename, img_draw)
        print(filename)
        count += 1

