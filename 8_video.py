# -*- coding: utf-8 -*-
import imageio
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files
from car.utils import plot_images

import cv2


START = 250
if __name__ == "__main__":
    img_files = list_files("project_video", pattern="*.jpg", random_order=False, recursive_option=False)    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    # d = ImgDetector(classifier=load_model("model_v4.pkl"))
    
    import numpy as np

    count = START
    for fname in img_files[START:]:
        img = cv2.imread(fname)

        count_str = "{}".format(count).zfill(5)
         
        filename = "..//debug//{}.txt".format(count_str)
        detect_boxes = np.loadtxt(filename).astype(int).reshape(-1,4)        
        img_draw = d.run(img, detect_boxes)
         
        filename = "..//debug//imgs//{}.jpg".format(count_str)
        cv2.imwrite(filename, img_draw)
        print(filename)
        count += 1





