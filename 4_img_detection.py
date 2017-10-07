# -*- coding: utf-8 -*-
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files
from car.utils import plot_images
from car.heatmap import HeatMap
import cv2


IMG_SRC_DIR = "test_images"

import matplotlib.pyplot as plt
def plot_images(images, titles):
    fig, ax = plt.subplots()
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(images), i+1)
        plt.title(titles[i])
        plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    img_files = list_files(IMG_SRC_DIR, pattern="*.jpg", random_order=False, recursive_option=False)    
    d = ImgDetector(classifier=load_model("model_v4.pkl"))
    heatmap_op = HeatMap()
    
    for fname in img_files:
        img = cv2.imread(fname)
        img_draw = d.run(img, (0, 400), do_heat_map=False)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        heatmap_op.show_process(img, d.detect_boxes)
#         img_draw_heat_map = d.run(img)
#         
#         plot_images([img, img_draw, img_draw_heat_map], ["input", "sliding window searched", "heat map operated"])
        break
