# -*- coding: utf-8 -*-

import cv2

class ImgScanner(object):
    
    def __init__(self, image, step_y=5, step_x=5, win_y=64, win_x=64):
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
            y : 
            x
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
     
    def __init__(self, image, scale=0.7, min_y=64, min_x=64):
        self.layer = image.copy()
        self.scale_for_original = 1.0
        
        self._scale = scale
        self._min_y = min_y
        self._min_x = min_x
     
    def generate_next(self):
        yield self.layer, self.scale_for_original
 
        while True:
            h = int(self.layer.shape[0] * self._scale)
            w = int(self.layer.shape[1] * self._scale)
             
            self.layer = cv2.resize(self.layer, (w, h))
             
            if h < self._min_y or w < self._min_x:
                break
            self.scale_for_original = self.scale_for_original * self._scale
            yield self.layer


class Scanner(object):
    
    def __init__(self, image):
        self._image = image
        self.layer = None
    
    def generate_next(self):
        image_pyramid = ImgPyramid(self._image)
        for layer in image_pyramid.generate_next():
            scanner = ImgScanner(layer)
            self.layer = layer
            for patch, y1, y2, x1, x2 in scanner.generate_next():
                y1_, y2_, x1_, x2_ = self._get_bb(y1, y2, x1, x2, image_pyramid.scale_for_original)
                yield patch, y1_, y2_, x1_, x2_
    
    def show_process(self):
        for patch, _, _, _, _ in self.generate_next():
            clone = self.layer.copy()
            cv2.rectangle(clone, (x, y), (x + self._win_x, y + self._win_y), (0, 255, 0), 2)
            cv2.imshow("Test Image Scanner", clone)
            cv2.waitKey(1)
            time.sleep(0.025)
        
    def _get_bb(self, y1, y2, x1, x2, scale_for_original):
        """Get bounding box in the original input image"""
        original_coords = [int(c / scale_for_original) for c in (y1, y2, x1, x2)]
        return original_coords


if __name__ == "__main__":
    import time
    
    image = cv2.imread("test_images//test1.jpg")[200:400, 200:400, :]
    scanner = ImgScanner(image)
    scanner.show_process()
    
    
    
    