# -*- coding: utf-8 -*-

import cv2
from car.train import load_model
from car.detect import ImgDetector


if __name__ == "__main__":
    
    img = cv2.imread("test_images//test1.jpg")
    d = ImgDetector(classifier=load_model("model.pkl"))
    img_draw = d.run(img, (0, 300))
    
    print(d.detect_boxes)
    cv2.imshow("Test Image Scanner", img_draw)
    cv2.waitKey(0)




