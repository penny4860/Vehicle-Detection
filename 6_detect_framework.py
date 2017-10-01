# -*- coding: utf-8 -*-

import cv2
from car.desc import get_hog_features
from car.train import load_model
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    from car.scan import Slider
    clf = load_model("model.pkl")
    
    img = cv2.imread("test_images//test1.jpg")[300:, :, :]
    slider = Slider(img)

    clone = img.copy()
    for patch in slider.generate_next():
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        feature_vector = get_hog_features([patch_gray])
        
        # predict_proba
        if clf.predict(feature_vector) == 1.0:
            p1, p2 = slider.get_bb()
            cv2.rectangle(clone, p1, p2, (0, 255, 0), 2)
            cv2.imshow("Test Image Scanner", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

