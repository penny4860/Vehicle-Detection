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
    imgs = files_to_images(img_files)[START:256]

    d = ImgDetector(classifier=load_model("model_v4.pkl"))

    import numpy as np    
    count = START
    for img in imgs:
        d.run(img)
        boxes = np.array(d.heat_boxes)
        count_str = "{}".format(count).zfill(5)
        filename = "..//debug//{}.txt".format(count_str)
        np.savetxt(filename, boxes, fmt='%d')
        print(filename)
        
        count += 1
    
