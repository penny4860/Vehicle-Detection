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


class HeatMap(object):
    def __init__(self, threshold=3):
        self._threshold = threshold
    
    def run(self, image, boxes):
        heat_map = np.zeros((image.shape[0], image.shape[1])).astype(float)
    
        for box in boxes:
            x1, y1, x2, y2 = box
            heat_map[y1:y2, x1:x2] += 1
            
        heat_map = self._get_bin(heat_map)
        return heat_map

    def _get_bin(self):
        heat_map_bin = np.zeros_like(heat_map)
        heat_map_bin[heat_map >= self._threshold] = 255
        return heat_map_bin

    def _set_boxes(self, heat_map):
        """
        # Args
            heat_map : ndarray
                binary image
        """
        pass

        
if __name__ == "__main__":
    from car.utils import plot_images
    import numpy as np
    
#     img = cv2.imread("..//test_images//test1.jpg")
#     d = ImgDetector(classifier=load_model("..//model.pkl"))
#     img_draw = d.run(img, (0, 300))
#     img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
#     heat_map = create_heat_map(img, d.detect_boxes)
#     np.save("heatmap", heat_map)

    heat_map = np.load("heatmap.npy")
    heat_map_bin = np.zeros_like(heat_map)
    heat_map_bin[heat_map >= 3] = 255
    
    from scipy.ndimage.measurements import label
    labels = label(heat_map_bin)
    print(labels[1])
    
#     # plot_images([heat_map, heat_map_bin])
#     import matplotlib.image as mpimg
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import cv2
#     
#     def draw_labeled_bboxes(img, labels):
#         # Iterate through all detected cars
#         for car_number in range(1, labels[1]+1):
#             # Find pixels with each car_number label value
#             nonzero = (labels[0] == car_number).nonzero()
#             # Identify x and y values of those pixels
#             nonzeroy = np.array(nonzero[0])
#             nonzerox = np.array(nonzero[1])
#             # Define a bounding box based on min/max x and y
#             bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
#             # Draw the box on the image
#             cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
#         # Return the image
#         return img
#     
#     # Read in the last image above
#     image = mpimg.imread('img105.jpg')
#     # Draw bounding boxes on a copy of the image
#     draw_img = draw_labeled_bboxes(np.copy(image), labels)
#     # Display the image
#     plt.imshow(draw_img)



