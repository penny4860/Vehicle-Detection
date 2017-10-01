# -*- coding: utf-8 -*-

import cv2
from car.train import load_model
from car.detect import ImgDetector


if __name__ == "__main__":
    
    img = cv2.imread("test_images//test1.jpg")
    d = ImgDetector(classifier=load_model("model.pkl"))
    img_draw = d.run(img[300:, :, :])
    print(len(d.detect_boxes))
    print(d.detect_boxes)
    
    import numpy as np   
    img_ = np.concatenate([img[:300, :, :], img_draw], axis=0)
    print(img_.shape)

    cv2.imshow("Test Image Scanner", img_)
    cv2.waitKey(0)




