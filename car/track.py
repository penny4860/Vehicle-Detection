
from filterpy.kalman import KalmanFilter
import numpy as np
np.set_printoptions(precision=2, suppress=True)

class Box(object):
    
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @classmethod
    def from_z(cls, px, py, scale, ratio):
        import math
        w = math.sqrt(scale * ratio)
        h = scale / w
        x1 = px - w/2
        x2 = px + w/2
        y1 = py - h/2
        y2 = py + h/2
        box = Box(x1, y1, x2, y2)
        return box

    def get_bb(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def get_z(self):
        z = np.array([self._px(), self._py(), self._scale(), self._ratio()]).reshape(-1,1)
        return z
    
    def _px(self):
        px = (self.x1 + self.x2) / 2 
        return px
    
    def _py(self):
        py = (self.y1 + self.y2) / 2
        return py

    def _scale(self):
        w = self.x2 - self.x1 
        h = self.y2 - self.y1
        scale = w*h
        return scale

    def _ratio(self):
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        ratio = float(w)/h
        return ratio

RELIABLE_THD = 20
DRAW_THD = 5
UNTRACK_THD = 5
class BoxTracker(object):
    
    _N_MEAS = 4         # (px, py, scale, ratio)-ordered
    _N_STATE = 7        # (px, py, scale, ratio, vx, vy, vs)-ordered
    
    def __init__(self, init_box, group_number=1):
        self._kf = self._build_kf(init_box)
        
        self.group_number = group_number
        self.detect_count = 1
        self.miss_count = 0

    def _build_kf(self, init_box, Q_scale=1.0, R_scale=400.0):
        kf = KalmanFilter(dim_x=self._N_STATE,
                          dim_z=self._N_MEAS)
        kf.F = np.array([[1,0,0,0,1,0,0],
                         [0,1,0,0,0,1,0],
                         [0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0],
                         [0,0,0,0,0,1,0],
                         [0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0],
                         [0,0,1,0,0,0,0],
                         [0,0,0,1,0,0,0]])
        Q = np.zeros_like(kf.F)
        for i in range(self._N_MEAS, self._N_STATE):
            Q[i, i] = Q_scale

        R = np.eye(self._N_MEAS) * R_scale
        
        kf.Q = Q
        kf.R = R
        
        kf.x = np.zeros((self._N_STATE, 1))
        kf.x[:self._N_MEAS,:] = init_box.get_z()
        return kf
    
    def predict(self):
        
        if self.is_missing_but_drawing():
            # v_scale
            self._kf.x[6,0] = 0 
        
        self._kf.predict()
        predict_box = Box.from_z(*self._kf.x[:self._N_MEAS,0])
        return predict_box
    
    def update(self, box=None):
        """
        # Args
            box : Box instance
                in the case of no matching box, box argument is None
        
        # Returns
            filtered_box : Box instance
        """
        if box is not None:
            self._detect_counting()
            z = box.get_z()
            self._kf.update(z)
            
        filtered_box = Box.from_z(*self._kf.x[:self._N_MEAS,0])
        return filtered_box
    
    def miss(self):
        self.miss_count += 1
    
    def get_bb(self):
        box = Box.from_z(*self._kf.x[:4,0])
        bounding_box = box.get_bb()
        return bounding_box

    def is_draw(self):
        if self._is_reliable_target() or self.detect_count-self.miss_count >= DRAW_THD:
            return True
        else:
            return False

    def is_delete(self):
        def _in_reliable_range():
            x1, y1, x2, y2 = self.get_bb()
            
            margin = 50
            # hard coding
            if x1 > margin and x2 < 1280-margin and y1 > 350 and y2 < 960-margin:
                return True
            else:
                return False
        
        if self.is_missing_but_drawing():
            if _in_reliable_range():
                return False
            else:
                return True
        else:
            if self.miss_count > self.detect_count or self.miss_count >= UNTRACK_THD:
                return True
            else:
                return False

    def is_missing_but_drawing(self):
        if self._is_reliable_target() and self.miss_count >= UNTRACK_THD:
            return True
        else:
            return False

    def _is_reliable_target(self):
        if self.detect_count >= RELIABLE_THD:
            return True
        else:
            return False

    def _detect_counting(self):
        self.detect_count += 1
        if self._is_reliable_target():
            self.miss_count = 0


if __name__ == "__main__":
    
    bbs = [(100, 100, 200, 200), (100, 100, 200, 200), (120, 120, 230, 230)]
    for i, bb in enumerate(bbs):
        box = Box(*bb)
        if i == 0:
            tracker = BoxTracker(box)
        filtered_box = tracker.run(box)
        
        print("==================================================================")
        print(box.get_bb())
        print(filtered_box.get_bb())
