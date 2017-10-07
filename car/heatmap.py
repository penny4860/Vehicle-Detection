# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label


def separate(boxes_):
    """separate heat boxes by aspect ratio"""
    def _separate_box(box, axis="x"):
        x1, y1, x2, y2 = box
        
        if axis == "x":
            px = (x1 + x2) / 2
            box1 = np.array([x1, y1, px, y2]).astype(int)
            box2 = np.array([px, y1, x2, y2]).astype(int)
        elif axis == "y":
            py = (y1 + y2) / 2
            box1 = np.array([x1, y1, x2, py]).astype(int)
            box2 = np.array([x1, py, x2, y2]).astype(int)
        return box1, box2
    
    boxes = np.array(boxes_).tolist()
    for box in boxes[:]:
        x1, y1, x2, y2 = box
        w = x2-x1
        h = y2-y1
        
        if w / h >= 1.85:
            print("separation x", w / h)
            box1, box2 = _separate_box(box, axis="x")
            boxes.remove(box)
            boxes.append(box1)
            boxes.append(box2)

        elif h / w >= 1.85:
            print("separation y")
            box1, box2 = _separate_box(box, axis="y")
            boxes.remove(box)
            boxes.append(box1)
            boxes.append(box2)
            
    return boxes


class HeatMap(object):
    def __init__(self, threshold=2):
        self._threshold = threshold
        self._heat_map = None
        self._heat_bin = None
        
    def get_boxes(self, boxes, w, h):
        """
        # Args
            boxes : list of tuple (x1, y1, x2, y2)
                detected boxes
            w : int
            h : int
        
        # Returns
            heat_boxes : list of tuple (x1, y1, x2, y2)
        """

        self._heat_map = np.zeros((h, w)).astype(float)
    
        for box in boxes:
            x1, y1, x2, y2 = box
            self._heat_map[y1:y2, x1:x2] += 1
            
        self._heat_bin = self._get_bin()
        heat_boxes = self._extract_boxes()
        return heat_boxes

    def show_process(self, image, boxes):
        """
        # Args
            image : 
                RGB-ordered original image
            boxes : list of tuple
                (x1, y1, x2, y2) ordered
        """
        def _plot_images(images, titles):
            _, axes = plt.subplots(1, len(images), figsize=(10,10))
            for img, ax, text in zip(images, axes, titles):
                ax.imshow(img, cmap="gray")
                ax.set_title(text, fontsize=30)
            plt.show()
            
        def _draw_box(image, boxes):
            drawn = image.copy()
            for box in boxes:
                cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), (0,0,255), 6)
            return drawn
            
        heat_boxes = self.get_boxes(boxes, image.shape[1], image.shape[0])
        original_box_img =_draw_box(image, boxes)
        heat_box_img = _draw_box(image, heat_boxes)

        _plot_images([self._heat_map, self._heat_bin, original_box_img, heat_box_img],
                     ["heat map", "thresholded heat map", "input boxes", "heat map processed boxes"])

    def _get_bin(self):
        heat_map_bin = np.zeros_like(self._heat_map)
        heat_map_bin[self._heat_map >= self._threshold] = 255
        return heat_map_bin

    def _extract_boxes(self):
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
        ccl_map, n_labels = label(self._heat_bin)
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
    d.run(img, (0, 300), False)

    heat_map = HeatMap()
    # boxes = heat_map.get_boxes(d.detect_boxes, img_draw.shape[1], img_draw.shape[0])
    heat_map.show_process(img, d.detect_boxes)
    
    