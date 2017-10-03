
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def match(cars, prev_cars):
    dist_map = np.zeros((len(cars), len(prev_cars)))
    for r, car in enumerate(cars):
        for c, p_car in enumerate(prev_cars):
            dist_map[r, c] = car.calc_dist(p_car)
    
    # print(dist_map)
    matching_pairs = linear_assignment(dist_map)  
    return matching_pairs   # (current_idx, prev_index)-ordered


DETECT_COUNTING = 2
class Car(object):
    def __init__(self, box):
        self._box = box
        self._detect_cnt = 1
        self._undetect_cnt = 0
        
    def calc_dist(self, a_car):
        point = np.array(self.get_point())
        a_point = np.array(a_car.get_point())
        dist = np.linalg.norm(point - a_point)
        return dist
        
    def get_point(self):
        """Get mid-point of the bounding box """
        x1, y1, x2, y2 = self._box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return x, y
 
    def detect_update(self, prev_car):
        # 1. box smoothing
        self._box = (2*np.array(self._box) + np.array(prev_car._box)) / 3
        
        # 2. detection count
        if self._detect_cnt <= DETECT_COUNTING:
            self._detect_cnt += 1

    def undetect_update(self):
        # 1. undetection count
        if self._undetect_cnt <= DETECT_COUNTING:
            self._undetect_cnt += 1

    def get_status(self):
        if self._detect_cnt > DETECT_COUNTING:
            return "detect"
        else:
            if self._undetect_cnt > DETECT_COUNTING:
                return "undetect"
            else:
                return "hold"
