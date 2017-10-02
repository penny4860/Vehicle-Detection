# -*- coding: utf-8 -*-

import cv2

from car.desc import get_hog_features
from car.train import load_model
from car.scan import MultipleScanner
from car.heatmap import HeatMap

class ImgDetector(object):
    
    def __init__(self, classifier, heat_map=HeatMap()):
        
        # Todo : Slider class 를 외부에서 주입받도록 수정하자.
        self._slider = None
        self._heat_map = heat_map
        self._clf = classifier
        
        self.detect_boxes = []
        self.heat_boxes = []
        
        self._start_x = 0
        self._start_y = 0
        
    
    def run(self, image, start_pt=(300,0), do_heat_map=True):
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

        layer = cv2.cvtColor(self._slider.layer, cv2.COLOR_RGB2GRAY)
        feature_map = get_hog_features([layer], feature_vector=False)
        
        for _ in self._slider.generate_next():
            if self._slider.layer.shape[0] != layer.shape[0]:
                layer = cv2.cvtColor(self._slider.layer, cv2.COLOR_RGB2GRAY)
                feature_map = get_hog_features([layer], feature_vector=False)
            
            feature_vector = self._get_feature_vector(feature_map)
            
            # predict_proba
            if self._clf.predict(feature_vector) == 1.0:
                self._set_detect_boxes()
        
        if do_heat_map:
            self.heat_boxes = self._heat_map.get_boxes(self.detect_boxes, image.shape[1], image.shape[0])
        else:
            self.heat_boxes = self.detect_boxes
        
        drawed = self._draw_boxes(image, self.heat_boxes)
        return drawed

    
    def _get_feature_vector(self, feature_map):
        pix_per_cell = 8
        cell_per_block=2
        unit = pix_per_cell - cell_per_block + 1
        
        p1, _ = self._slider.get_pyramid_bb()
        x1 = p1[0]//pix_per_cell
        y1 = p1[1]//pix_per_cell
        x2 = x1 + unit
        y2 = y1 + unit
        feature_vector = feature_map[:, y1:y2, x1:x2, :, :, :].ravel()
        feature_vector = feature_vector.reshape(1, -1)
        return feature_vector

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

    def _draw_boxes(self, image, boxes):
        """Draw detected boxes to an image"""
        
        clone = image.copy()
        for box in boxes:
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(clone, p1, p2, (0, 255, 0), 2)
        return clone

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = plt.imread("..//test_images//test1.jpg")
    d = ImgDetector(classifier=load_model("..//model.pkl"))
    drawn = d.run(img, (0, 300))

    plt.imshow(drawn)
    plt.show()


