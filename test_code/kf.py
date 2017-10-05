
from filterpy.kalman import KalmanFilter
import numpy as np


class Box(object):
    
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def get_bb(self):
        return self.x1, self.y1, self.x2, self.y2
    
    def get_point(self):
        px = (self.x1 + self.x2) / 2 
        py = (self.y1 + self.y2) / 2
        return px, py

    def get_scale(self):
        w = self.x2 - self.x1 
        h = self.y2 - self.y1
        scale = w*h
        return scale

    def get_ratio(self):
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        ratio = float(w)/h
        return ratio

class BoxTracker(object):
    
    _N_MEAS = 4         # (px, py, scale, ratio)
    _N_STATE = 7        # (px, py, scale, ratio, vx, vy, vs)
    
    def __init__(self):
        self._kf = self._build_kf()
    
    def _build_kf(self):
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
        Q_scale = 0.01
        R_scale = 10.0
        Q = np.zeros_like(kf.F)
        Q[self._N_MEAS:, self._N_MEAS:] = Q_scale
        R = np.eye(self._N_MEAS) * R_scale
        
        kf.Q = Q
        kf.R = R
        # kf.x = np.array([0, 0, 0, 0]).reshape(-1,1)
        return kf
    
    def run(self, measured):
        """
        # Args
            measured : array
                (x, y, w*h, w/h)-ordered
        
        # Returns
            output : array
                (x, y, w*h, w/h)-ordered
        
        """
        z = np.array(measured).reshape(-1, 1)

        self._kf.predict()
        # predict_state = self._kf.x
        
        self._kf.update(z)
        # posterior_state = self._kf.x
        output = self._kf.x[0:4]
        return output

#     for meas in zip(observed_x, observed_y):
#         kf.predict()
#         z = np.array([meas[0], meas[1]]).reshape(-1,1)
#         kf.update(z)
#         result.append(kf.x[0:2])
#         print(meas, kf.x[0], kf.x[1])


# import matplotlib.pyplot as plt
# def demo_kalman_xy():
#     N = 100
#     true_x = np.linspace(0.0, 1000.0, N)
#     true_y = true_x
#     observed_x = true_x + 10*np.random.random(N)
#     observed_y = true_y + 10*np.random.random(N)
#     plt.plot(observed_x, observed_y, 'ro')
#     result = []
# 
#     
#     for meas in zip(observed_x, observed_y):
#         kf.predict()
#         z = np.array([meas[0], meas[1]]).reshape(-1,1)
#         kf.update(z)
#         result.append(kf.x[0:2])
#         print(meas, kf.x[0], kf.x[1])
#         
#     kalman_x, kalman_y = zip(*result)
#     plt.plot(kalman_x, kalman_y, 'g-')
#     plt.show()
#  
# demo_kalman_xy()
