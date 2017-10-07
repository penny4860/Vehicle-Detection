# -*- coding: utf-8 -*-

import cv2

from car.desc import HogDesc, HogMap
from car.train import load_model
from car.scan import MultipleScanner
from car.heatmap import HeatMap, separate

# Todo : Box package #############################################
from car.track import BoxTracker, Box
from car.match import BoxMatcher
##################################################################

import numpy as np


MAX_TRACKERS = 10
class VideoDetector(object):
    def __init__(self, img_detector):
        self._img_detector = img_detector

        # BoxTracker instances
        self._box_trackers = []
        self._group_idxes = np.array([False]*MAX_TRACKERS)
    
    def _detect(self, img):
        def _is_obscured():
            is_exist = False
            for tracker in self._box_trackers:
                if tracker.is_missing_but_drawing():
                    is_exist = True
            return is_exist
        
        _ = self._img_detector.run(img, do_heat_map=True)

        detected_boxes = self._img_detector.heat_boxes
        if _is_obscured():
            detected_boxes = separate(detected_boxes)
        
        return detected_boxes

    def _get_pred_boxes(self):
        tracking_boxes = []

        for tracker in self._box_trackers:
            tracker.predict()
            tracking_boxes.append(tracker.get_bb())
            
        tracking_boxes = np.array(tracking_boxes)
        return tracking_boxes

    def _assign_group_index(self):
        idx =  np.where(self._group_idxes == False)[0][0]
        self._group_idxes[idx] = True
        return idx
        
    def run(self, img, draw_unfiltered_box=True):

        # 1. run still image detection framework
        detect_boxes = np.array(self._detect(img))

        # 2. get tracking boxes
        tracking_boxes = self._get_pred_boxes()

        # 3. matching 2-list of boxes
        box_matcher = BoxMatcher(detect_boxes, tracking_boxes)
        
        # 4. detected boxes op
        new_box_trackers = []
        for i, _ in enumerate(detect_boxes):
            tracking_idx, iou = box_matcher.match_idx_of_box1_idx(i)
            # Todo : iou 로 thresholding
            
            if tracking_idx is None:
                # create new tracker
                box_tracker = BoxTracker(Box(*detect_boxes[i]), self._assign_group_index())
                new_box_trackers.append(box_tracker)
            else:
                # run tracker by measured detection box
                measurement_box = Box(*detect_boxes[i])
                self._box_trackers[tracking_idx].update(measurement_box)

        # 5. tracking but unmatched traker process
        for i, _ in enumerate(self._box_trackers):
            idx, iou = box_matcher.match_idx_of_box2_idx(i)

            # missing target
            if idx is None:
                self._box_trackers[i].miss()

        self._box_trackers += new_box_trackers

        # 6. delete tracker in trackers
        for tracker in self._box_trackers[:]:
            if tracker.is_delete():
                self._group_idxes[tracker.group_number] = False
                self._box_trackers.remove(tracker)

        if draw_unfiltered_box:
            img_clone = self._draw_boxes(img, detect_boxes, (255, 0, 0), 8)
        else:
            img_clone = img.copy()
        
        # 7. draw box
        for tracker in self._box_trackers:
            if tracker.is_draw():
                img_clone = self._draw_group_box(img_clone, tracker.get_bb(), tracker.group_number)
                
        return img_clone

    def _draw_group_box(self, image, box, g_number, color=(0, 255, 0), thickess=4):
        clone = image.copy()

        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        cv2.rectangle(clone, p1, p2, color, thickess)
        cv2.putText(clone, "car:{}".format(g_number), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
        return clone
    
    def _draw_boxes(self, image, boxes, color=(0, 255, 0), thickess=2):
        """Draw detected boxes to an image"""
         
        clone = image.copy()
        for _, box in enumerate(boxes):
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(clone, p1, p2, color, thickess)
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
         
     
    def run(self, image, start_pt=(640,400), end_pt=(1280, 400+256), do_heat_map=True):
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
    pass


