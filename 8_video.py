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


if __name__ == "__main__":
    
    img_files = list_files("video", pattern="*.jpg", random_order=False)
    imgs = files_to_images(img_files)
    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v3.pkl")))
    
    count = 0
    for img in imgs:
        img_draw = d.run(img)
        
        count_str = "{}".format(count).zfill(5)
        filename = "video//img_detect_{}.jpg".format(count_str)
        cv2.imwrite(filename, img_draw)
        print(filename)
        count += 1

