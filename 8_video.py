# -*- coding: utf-8 -*-
import imageio
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files
from car.utils import plot_images

import cv2


def get_boxes_from_img_frm(filename):
    detect_boxes = np.loadtxt(filename).astype(int).reshape(-1,4)        
    return detect_boxes

def run_video_frm(fname, detect_boxes, count_str):
    img = cv2.imread(fname)
    img_draw = d.run(img, detect_boxes)
    filename = "..//debug//imgs//{}.jpg".format(count_str)
    cv2.imwrite(filename, img_draw)
    print(filename)


START = 250
if __name__ == "__main__":
    img_files = list_files("..//project_video", pattern="*.jpg", random_order=False, recursive_option=False)    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    # d = ImgDetector(classifier=load_model("model_v4.pkl"))
    
    import numpy as np

    count = START
    for fname in img_files[START:995]:
        count_str = "{}".format(count).zfill(5)
        filename = "..//debug//{}.txt".format(count_str)
        
        # 1. get boxes from text file
        detect_boxes = get_boxes_from_img_frm(filename)

        # 2. run test
        run_video_frm(fname, detect_boxes, count_str)
        count += 1





