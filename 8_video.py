# -*- coding: utf-8 -*-
import imageio
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files
from car.utils import plot_images

import cv2


def files_to_images(files):
    import numpy as np
    images = []
    for filename in files:
        image = cv2.imread(filename)
        images.append(image)
    images = np.array(images)
    return images

START = 250
if __name__ == "__main__":
    img_files = list_files("project_video", pattern="*.jpg", random_order=False, recursive_option=False)
    imgs = files_to_images(img_files)[START:]
    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    # d = ImgDetector(classifier=load_model("model_v4.pkl"))
    
    count = START
    for img in imgs:
        img_draw = d.run(img)
        
        count_str = "{}".format(count).zfill(5)
        filename = "project_video//debug4//{}.jpg".format(count_str)
        cv2.imwrite(filename, img_draw)
        print(filename)
        count += 1

