
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


DRAW_THD = 3
UNTRACK_THD = 5
class BoxTracker(object):
    
    _N_MEAS = 4         # (px, py, scale, ratio)-ordered
    _N_STATE = 7        # (px, py, scale, ratio, vx, vy, vs)-ordered
    
    def __init__(self, init_box, group_number=1):
        self._kf = self._build_kf(init_box)
        
        self.group_number = group_number
        self.detect_count = 1
        self.miss_count = 0

    def _build_kf(self, init_box, Q_scale=0.01, R_scale=10.0):
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
        Q[self._N_MEAS:, self._N_MEAS:] = Q_scale
        R = np.eye(self._N_MEAS) * R_scale
        
        kf.Q = Q
        kf.R = R
        
        kf.x = np.zeros((self._N_STATE, 1))
        kf.x[:4,:] = init_box.get_z()
        return kf
    
    def predict(self):
        self._kf.predict()
        predict_box = Box.from_z(*self._kf.x[:4,0])
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
            self.detect_count += 1
            z = box.get_z()
            self._kf.update(z)
            
        filtered_box = Box.from_z(*self._kf.x[:4,0])
        return filtered_box
    
    def miss(self):
        self.miss_count += 1
    
    def get_bb(self):
        box = Box.from_z(*self._kf.x[:4,0])
        bounding_box = box.get_bb()
        return bounding_box

    def is_draw(self):
        if self.detect_count >= DRAW_THD:
            return True

    def is_delete(self):
        if self.miss_count >= UNTRACK_THD:
            return True


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
        
        
        
    
    
