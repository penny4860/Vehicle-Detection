# -*- coding: utf-8 -*-

import cv2

from car.desc import get_hog_features
from car.train import load_model
from car.scan import Slider


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
                BGR-ordered image
            start_pt : tuple
                (x, y)

        # Returns
            drawed : ndarray, same size of image
                Image with patch recognized in input image        
        """
        self._start_x = start_pt[0]
        self._start_y = start_pt[1]
        
        scan_img = image[start_pt[1]:, start_pt[0]:, :]
        
        self._slider = Slider(scan_img)
        for patch in self._slider.generate_next():
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
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

from scipy.ndimage.measurements import label
class HeatMap(object):
    def __init__(self, threshold=3):
        self._threshold = threshold
    
    def get_boxes(self, boxes, w, h):
        heat_map = np.zeros((h, w)).astype(float)
    
        for box in boxes:
            x1, y1, x2, y2 = box
            heat_map[y1:y2, x1:x2] += 1
            
        heat_map_bin = self._get_bin(heat_map)
        heat_map_boxes = self._extract_boxes(heat_map_bin)
        return heat_map_boxes

    def _get_bin(self, heat_map):
        heat_map_bin = np.zeros_like(heat_map)
        heat_map_bin[heat_map >= self._threshold] = 255
        return heat_map_bin

    def _extract_boxes(self, heat_map_bin):
        """
        # Args
            heat_map : ndarray
                binary image
        """
        def _box(ccl_map, car_number):
            # Find pixels with each car_number label value
            nonzero = (ccl_map == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            x1 = np.min(nonzerox)
            y1 = np.min(nonzeroy)
            x2 = np.max(nonzerox)
            y2 = np.max(nonzeroy)
            return (x1, y1, x2, y2)
            
        boxes = []
        ccl_map, n_labels = label(heat_map_bin)
        # Iterate through all detected carsccl_map[1]
        for car_number in range(1, n_labels+1):
            box = _box(ccl_map, car_number)
            boxes.append(box)
        return boxes

        
if __name__ == "__main__":
    from car.utils import plot_images
    import numpy as np
    
    img = cv2.imread("..//test_images//test1.jpg")
    d = ImgDetector(classifier=load_model("..//model.pkl"))
    img_draw = d.run(img, (0, 300))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    
    heat_map = HeatMap()
    boxes = heat_map.get_boxes(d.detect_boxes, img_draw.shape[1], img_draw.shape[0])
    
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 6)

    plot_images([img, img_draw])
     
