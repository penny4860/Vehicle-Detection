# -*- coding: utf-8 -*-

import cv2
from car.scan import Slider

if __name__ == "__main__":
    image = cv2.imread("test_images//test1.jpg")[200:400, 200:400, :]
    slider = Slider(image)
    slider.show_process()
    
    
    