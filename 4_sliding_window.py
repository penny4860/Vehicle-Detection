# -*- coding: utf-8 -*-

import cv2
from car.scan import MultipleScanner

if __name__ == "__main__":
    image = cv2.imread("test_images//test1.jpg")[200:400, 200:400, :]
    slider = MultipleScanner(image)
    slider.show_process()
    
    
    