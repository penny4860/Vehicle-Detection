# -*- coding: utf-8 -*-

import cv2
import time

class ImgScanner(object):
    
    def __init__(self, image, step_y=16, step_x=16, win_y=64, win_x=64):
        self._layer = image
        self._step_x = step_x
        self._step_y = step_y
        self._win_x = win_x
        self._win_y = win_y
        
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        
    def generate_next(self):
        """Generate next patch
        
        # Yields
            patch : ndarray, shape of (self._win_y, self._win_x) or (self._win_y, self._win_x, 3)
        """
        
        for y in range(0, self._layer.shape[0] - self._win_y, self._step_y):
            for x in range(0, self._layer.shape[1] - self._win_x, self._step_x):
                self.y1 = y
                self.y2 = y + self._win_y
                self.x1 = x
                self.x2 = x + self._win_x
                patch = self._layer[self.y1:self.y2, self.x1:self.x2]
                yield patch

    def get_bb(self):
        p1 = (self.x1, self.y1)
        p2 = (self.x2, self.y2)
        return p1, p2
        
    def get_patches(self):
        patches = [patch for patch in self.generate_next()]
        return patches

    def show_process(self):
        for _ in self.generate_next():
            clone = self._layer.copy()
            p1, p2 = self.get_bb()
            cv2.rectangle(clone, p1, p2, (0, 255, 0), 2)
            cv2.imshow("Test Image Scanner", clone)
            cv2.waitKey(1)
            time.sleep(0.025)


class ImgPyramid(object):
     
    def __init__(self, image, scale=0.8, min_y=96, min_x=96):
        self.layer = image.copy()
        self.scale_for_original = 1.0
        
        self._scale = scale
        
        self._min_y = min_y
        self._min_x = min_x
     
    def generate_next(self):
        # yield self.layer
 
        while True:
            h = int(self.layer.shape[0] * self._scale)
            w = int(self.layer.shape[1] * self._scale)
             
            self.layer = cv2.resize(self.layer, (w, h))
             
            if h < self._min_y or w < self._min_x:
                break
            self.scale_for_original = self.scale_for_original * self._scale
            yield self.layer

    def show_process(self):
        for _ in self.generate_next():
            clone = self.layer.copy()
            cv2.imshow("Test Image Pyramid", clone)
            cv2.waitKey(1)
            time.sleep(0.25)


class MultipleScanner(object):
    
    def __init__(self, image):
        self._image = image
        self.layer = None
        
        self.img_scanner = None
        self.img_pyramid = None
        self._updated = False

    def generate_next(self):
        self.img_pyramid = ImgPyramid(self._image)
        for layer in self.img_pyramid.generate_next():
            self.img_scanner = ImgScanner(layer)
            self.layer = layer
            self._updated = True

            for patch in self.img_scanner.generate_next():
                p1, p2 = self.img_scanner.get_bb()
                self._set_original_box(p1, p2)
                yield patch
                self._updated = False

    def get_bb(self):
        """Get coordinates being scanned in the original image"""
        p1 = (self._x1, self._y1)
        p2 = (self._x2, self._y2)
        return p1, p2

    def get_pyramid_bb(self):
        """Get coordinates being scanned in the scaled layer"""
        p1, p2 = self.img_scanner.get_bb()
        return p1, p2

    def is_updated_layer(self):
        return self._updated
    
    def show_process(self):
        for _ in self.generate_next():
            clone = self._image.copy()
            p1, p2 = self.get_bb()
            cv2.rectangle(clone, p1, p2, (0, 255, 0), 2)
            cv2.imshow("Test Image Scanner", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

    def _set_original_box(self, p1, p2):
        """Set bounding box coordinate in the original image"""
        p1_original = [int(c / self.img_pyramid.scale_for_original) for c in (p1)]
        p2_original = [int(c / self.img_pyramid.scale_for_original) for c in (p2)]

        self._x1, self._y1 = p1_original
        self._x2, self._y2 = p2_original


if __name__ == "__main__":
    image = cv2.imread("..//test_images//test1.jpg")[400:656, :, :]
    slider = MultipleScanner(image)
    slider.show_process()
    
    
    