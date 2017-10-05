# -*- coding: utf-8 -*-

import cv2
from car.train import load_model
from car.detect import ImgDetector

if __name__ == "__main__":
    
    img = cv2.imread("project_video//00642.jpg")
    d = ImgDetector(classifier=load_model("model_v4.pkl"))
    img_draw = d.run(img, start_pt=(0,350), end_pt=(1280, 400+256), do_heat_map=True)
    
    print(d.detect_boxes)
    print(d.heat_boxes)
    
    import matplotlib.pyplot as plt
    plt.imshow(img_draw)
    plt.show()
    
#     cv2.imshow("Test Image Scanner", img_draw)
#     cv2.waitKey(0)




