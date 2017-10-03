# -*- coding: utf-8 -*-
import imageio
from car.train import load_model
from car.detect import ImgDetector, VideoDetector
from car.data import list_files, files_to_images
from car.utils import plot_images


if __name__ == "__main__":
    
    img_files = list_files("video", pattern="*.jpg", random_order=False)
    imgs = files_to_images(img_files)


    d = VideoDetector(ImgDetector(classifier=load_model("model_v3.pkl")))
    for img in imgs:
        img_draw = d.run(img)
        plot_images(img_draw)
