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
    files = ["project_video//00642.jpg",
             "project_video//00643.jpg",
             "project_video//00644.jpg",
             "project_video//00645.jpg",
             "project_video//00646.jpg",
             "project_video//00647.jpg",
             "project_video//00648.jpg"]
    
    d = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    # d = ImgDetector(classifier=load_model("model_v4.pkl"))
    
    img_detected = []
    count = 0
    for fname in files:
        img = cv2.imread(fname)
        img_draw = d.run(img)
        plot_images([img, img_draw])
        
        
#         count_str = "{}".format(count).zfill(5)
#         filename = "{}.jpg".format(count_str)
#         cv2.imwrite(filename, img_draw)
#         print(filename)
#         count += 1

