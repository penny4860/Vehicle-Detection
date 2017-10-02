# -*- coding: utf-8 -*-

import cv2

from car.desc import get_hog_features
from car.train import load_model
from car.scan import MultipleScanner


class ImgDetector(object):
    
    def __init__(self, classifier):
        self._slider = None
        self._clf = classifier
        self.detect_boxes = []
        
        self._start_x = 0
        self._start_y = 0
        
    
    def run(self, image, start_pt=(300,0)):
        """
        # Args
            image : ndarray, shape of (H, W, 3)
                RGB-ordered image
            start_pt : tuple
                (x, y)

        # Returns
            drawed : ndarray, same size of image
                Image with patch recognized in input image        
        """
        self._start_x = start_pt[0]
        self._start_y = start_pt[1]
        
        scan_img = image[start_pt[1]:, start_pt[0]:, :]
        
        self._slider = MultipleScanner(scan_img)
        for patch in self._slider.generate_next():
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            
            # Todo : get_hog_features -> class
            feature_vector = get_hog_features([patch_gray])
            
            # predict_proba
            if self._clf.predict(feature_vector) == 1.0:
                self._set_detect_boxes()
        drawed = self._draw_boxes(image)
        return drawed

    def _set_detect_boxes(self):
        """Set detected box coordinate"""
        # Get current box coordinate & Draw
        p1, p2 = self._slider.get_bb()
        x1, y1 = p1
        x2, y2 = p2
        box = (x1 + self._start_x,
               y1 + self._start_y,
               x2 + self._start_x,
               y2 + self._start_y)
        self.detect_boxes.append(box)

    def _draw_boxes(self, image):
        """Draw detected boxes to an image"""
        
        clone = image.copy()
        for box in self.detect_boxes:
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(clone, p1, p2, (0, 255, 0), 2)
        return clone

        
if __name__ == "__main__":
    pass
