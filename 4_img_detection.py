# -*- coding: utf-8 -*-
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files
from car.utils import plot_images

import cv2


IMG_SRC_DIR = "test_images"


if __name__ == "__main__":
    img_files = list_files(IMG_SRC_DIR, pattern="*.jpg", random_order=False, recursive_option=False)    
    d = ImgDetector(classifier=load_model("model_v4.pkl"))
    
    for fname in img_files:
        img = cv2.imread(fname)
        img_draw = d.run(img)
        cv2.imshow("{} : image detection framework".format(fname), img_draw)
        cv2.waitKey(0)
