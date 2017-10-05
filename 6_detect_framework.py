# -*- coding: utf-8 -*-

import cv2
from car.train import load_model
from car.detect import ImgDetector


if __name__ == "__main__":
    
    img = cv2.imread("project_video//00640.jpg")
    img = cv2.imread("project_video//00700.jpg")
    d = ImgDetector(classifier=load_model("model_v4.pkl"))
    img_draw = d.run(img, do_heat_map=False)
    
    print(d.detect_boxes)
    cv2.imshow("Test Image Scanner", img_draw)
    cv2.waitKey(0)




