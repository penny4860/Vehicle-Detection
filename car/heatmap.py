# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label

class HeatMap(object):
    def __init__(self, threshold=3):
        self._threshold = threshold
    
    def get_boxes(self, boxes, w, h, plot=False):
        """
        # Args
            boxes : list of tuple (x1, y1, x2, y2)
                detected boxes
            w : int
            h : int
        
        # Returns
            heat_map_boxes : list of tuple (x1, y1, x2, y2)
        """

        heat_map = np.zeros((h, w)).astype(float)
    
        for box in boxes:
            x1, y1, x2, y2 = box
            heat_map[y1:y2, x1:x2] += 1
            
        heat_map_bin = self._get_bin(heat_map)
        heat_map_boxes = self._extract_boxes(heat_map_bin)
        
        if plot:
            self._show_process(heat_map, heat_map_bin, heat_map_boxes)
        
        return heat_map_boxes

    def _show_process(self, heat_map, bin_map, boxes):
        def _plot_images(images, titles):
            _, axes = plt.subplots(1, len(images), figsize=(10,10))
            for img, ax, text in zip(images, axes, titles):
                ax.imshow(img, cmap="gray")
                ax.set_title(text, fontsize=30)
            plt.show()

        drawn = np.zeros_like(heat_map)
        for box in boxes:
            cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), (0,0,255), 6)

        _plot_images([heat_map, bin_map, drawn], ["heat map", "thresholded heat map", "extracted boxes"])

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


if __name__ == '__main__':
    from car.detect import ImgDetector
    from car.train import load_model
    from car.utils import plot_images

    img = plt.imread("..//test_images//test1.jpg")
    d = ImgDetector(classifier=load_model("..//model.pkl"))
    img_draw = d.run(img, (0, 300))

    heat_map = HeatMap()
    boxes = heat_map.get_boxes(d.detect_boxes, img_draw.shape[1], img_draw.shape[0])
    
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 6)

    plot_images([img, img_draw])
