# -*- coding: utf-8 -*-

import cv2

from car.desc import HogDesc, HogMap
from car.train import load_model
from car.scan import MultipleScanner
from car.heatmap import HeatMap
from car.car import Car, match

import numpy as np

 
class VideoDetector(object):
    def __init__(self, img_detector):
        self._img_detector = img_detector
        self._prev_cars = []
        
    def _create_cur_cars(self, heat_boxes):
        cur_cars = []
        for box in heat_boxes:
            cur_cars.append(Car(box))
        return cur_cars

    def run(self, img):
        _ = self._img_detector.run(img, do_heat_map=True)
        
        # 1. 인식된 객체에 대한 list 생성
        cars = self._create_cur_cars(self._img_detector.heat_boxes)
        print("heat boxes", len(cars), end=", ")
        
        # 2. 이전 frame 과 matching
        pairs = match(cars, self._prev_cars) # [ (현재, 과거), .... ]
        
        # 3. matching 된 현재 frame 에서의 box를 처리
        for i, cur_car in enumerate(cars):
            # matched
            if i in pairs[:, 0]:
                matching_idx = np.where(pairs[:, 0] == i)[0][0]
                prev_idx = int(pairs[matching_idx, 1])
                cur_car.detect_update(self._prev_cars[prev_idx])

        # 4. unmatching 된 과거 frame 에서의 box를 처리
        for i, prev_car in enumerate(self._prev_cars):
            # unmatched
            if i not in pairs[:, 1]:
                prev_car.undetect_update()
                if prev_car.get_status() == "hold":
                    cars.append(prev_car)
        
        print("cur cars", len(cars), end=", ")
        boxes = []
        for car in cars:
            if car.get_status() == "detect":
                boxes.append(car.get_box())
        print("draw boxes", len(boxes), end=", ")
        
        import copy
        self._prev_cars = copy.deepcopy(cars)
        
        # 
        clone = self._draw_boxes(img, self._img_detector.heat_boxes, (255, 0, 0))
        return self._draw_boxes(clone, boxes)


    def _draw_boxes(self, image, boxes, color=(0, 255, 0)):
        """Draw detected boxes to an image"""
         
        clone = image.copy()
        for box in boxes:
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(clone, p1, p2, color, 2)
        return clone

 
class ImgDetector(object):
     
    def __init__(self, classifier, heat_map=HeatMap()):
         
        # Todo : Slider class 를 외부에서 주입받도록 수정하자.
        self._slider = None
        self._heat_map = heat_map
        self._clf = classifier
         
        # Todo : 외부에서 주입
        self._desc_map = HogMap(HogDesc())
         
        self.detect_boxes = []
        self.heat_boxes = []
        self._feature_map = None
         
        self._start_x = 0
        self._start_y = 0
         
     
    def run(self, image, start_pt=(0,400), end_pt=(1280, 400+256), do_heat_map=True):
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
        self.detect_boxes = []
        self.heat_boxes = []
         
        # 1. Run offset handling operation
        scan_img = self._run_offset(image, start_pt, end_pt)
 
        # 2. Multiple sized sliding window scanner
        self._slider = MultipleScanner(scan_img)
        for _ in self._slider.generate_next():
             
            # 3. Get feature vector and run classifier
            feature_vector = self._get_feature_vector()
            if self._clf.predict(feature_vector) == 1.0:
                self._set_detect_boxes()
 
        # 4. Run heat map operation        
        if do_heat_map:
            self.heat_boxes = self._heat_map.get_boxes(self.detect_boxes, image.shape[1], image.shape[0])
        else:
            self.heat_boxes = self.detect_boxes
         
        # 5. Draw detected boxes in the input image & return it
        drawed = self._draw_boxes(image, self.heat_boxes)
        return drawed
 
    def _run_offset(self, image, start_pt, end_pt):
        self._start_x = start_pt[0]
        self._start_y = start_pt[1]
         
        if end_pt is not None:
            return image[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        else:
            return image[start_pt[1]:, start_pt[0]:, :]
     
    def _get_feature_vector(self):
        if self._slider.is_updated_layer():
            layer = cv2.cvtColor(self._slider.layer, cv2.COLOR_RGB2GRAY)
            self._desc_map.set_features(layer)
 
        start_pt, _ = self._slider.get_pyramid_bb()
        feature_vector = self._desc_map.get_features(start_pt[0], start_pt[1])
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
    prev_cars = [Car([500,500, 664, 664]), Car([100,100, 164, 164]), Car([300,300, 364, 364])]
    cars = [Car([110,110, 174, 174]), Car([310,310, 374, 374])]
    print(match(cars, prev_cars))


