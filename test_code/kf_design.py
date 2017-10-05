# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

class _BaseKalmanDesigner(object):
    
    def __init__(self, init_pos, P, Q_std, R_std, dt=1.0, w=None):
        
        self._init_pos = init_pos
        self._P = P
        self._Q_std = Q_std
        self._R_std = R_std
        self._dt = dt
        self._w = w
    
    # todo : abstract method 로 만들자.
    def design(self):
        pass


class ConstTurnDesigner(_BaseKalmanDesigner):
    
    _n_state_vars = 6
    _n_measure_vars = 2
    
    def design(self):
        dt = self._dt
        w = self._w
        
        # 1. Choose the State Variables (initial velocity = 0)
        x0 = np.array([[self._init_pos[0]],
                       [0.0],
                       [0.0],
                       [self._init_pos[1]],
                       [0.0],
                       [0.0]])
        
        # 2. Design State Transition Function
        F = np.array([[1,             dt, 0,  0,    -w*dt**2/2,  0],
                      [0,  1-(w*dt)**2/2, 0,  0,         -w*dt,  0],
                      [0,              0, 0,  0,             0,  0],
                      [0,      w*dt**2/2, 0,  1,            dt,  0],
                      [0,           w*dt, 0,  0, 1-(w*dt)**2/2,  0],
                      [0,              0, 0,  0,             0,  0]])
        
        # 3. Design the Process Noise Matrix
        Q = np.zeros_like(F)
        Q[0,0] = self._Q_std ** 2
        Q[1,1] = self._Q_std ** 2
        Q[3,3] = self._Q_std ** 2
        Q[4,4] = self._Q_std ** 2
        
        #4. Control
        #5. Design the Measurement Function
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])
        #6. Measurement Noise Matrix
        R = np.eye(self._n_measure_vars) * self._R_std**2
        
        #7. Initial Condition
        if self._P is None:
            self._P = np.eye(self._n_state_vars) * 1.0

        return F, H, x0, self._P, Q, R


class ConstVelDesigner(_BaseKalmanDesigner):
    
    _n_state_vars = 6
    _n_measure_vars = 2
    
    def design(self):
        dt = self._dt
        
        # 1. Choose the State Variables (initial velocity = 0)
        x0 = np.array([[self._init_pos[0]],
                       [0.0],
                       [0.0],
                       [self._init_pos[1]],
                       [0.0],
                       [0.0]])
        
        # 2. Design State Transition Function
        F = np.array([[1, dt, 0,  0, 0,  0],
                      [0,  1, 0,  0, 0,  0],
                      [0,  0, 0,  0, 0,  0],
                      [0,  0, 0,  1, dt, 0],
                      [0,  0, 0,  0, 1,  0],
                      [0,  0, 0,  0, 0,  0]])
        
        # 3. Design the Process Noise Matrix
        q = Q_discrete_white_noise(dim=self._n_state_vars/2, 
                                   dt=dt, 
                                   var=self._Q_std**2)
        Q = np.zeros_like(F)
        Q[1,1] = self._Q_std**2
        Q[4,4] = self._Q_std**2

        #4. Control
        #5. Design the Measurement Function
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])
        #6. Measurement Noise Matrix
        R = np.eye(self._n_measure_vars) * self._R_std**2
        
        #7. Initial Condition
        if self._P is None:
            self._P = np.eye(self._n_state_vars) * 1.0

        return F, H, x0, self._P, Q, R

class StationaryDesigner(_BaseKalmanDesigner):
    
    _n_state_vars = 6
    _n_measure_vars = 2
    
    def design(self):
        dt = self._dt
        
        # 1. Choose the State Variables (initial velocity = 0)
        x0 = np.array([[self._init_pos[0]],
                       [0.0],
                       [0.0],
                       [self._init_pos[1]],
                       [0.0],
                       [0.0]])
        
        # 2. Design State Transition Function
        F = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        
        # 3. Design the Process Noise Matrix
        q = Q_discrete_white_noise(dim=self._n_state_vars/2, 
                                   dt=dt, 
                                   var=self._Q_std**2)
        Q = np.zeros_like(F)
        Q[0,0] = self._Q_std**2
        Q[3,3] = self._Q_std**2
        
        #4. Control
        #5. Design the Measurement Function
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])
        #6. Measurement Noise Matrix
        R = np.eye(self._n_measure_vars) * self._R_std**2
        
        #7. Initial Condition
        if self._P is None:
            self._P = np.eye(self._n_state_vars) * 1.0

        return F, H, x0, self._P, Q, R

class ConstAccDesigner(_BaseKalmanDesigner):
    
    _n_state_vars = 6
    _n_measure_vars = 2
    
    def design(self):
        dt = self._dt
        
        # 1. Choose the State Variables (initial velocity = 0, initial_acc = 0)
        x0 = np.array([[self._init_pos[0]],
                       [0.0],
                       [0.0],
                       [self._init_pos[1]],
                       [0.0],
                       [0.0]])
        
        # 2. Design State Transition Function (6x6)
        F = np.array([[1, dt, 0.5*dt**2, 0,  0,         0],
                      [0,  1,        dt, 0,  0,         0],
                      [0,  0,         1, 0,  0,         0],
                      [0,  0,         0, 1, dt, 0.5*dt**2],
                      [0,  0,         0, 0,  1,        dt],
                      [0,  0,         0, 0,  0,         1]])

        # 3. Design the Process Noise Matrix
        q = Q_discrete_white_noise(dim=self._n_state_vars/2, 
                                   dt=dt, 
                                   var=self._Q_std**2)
        Q = np.zeros_like(F)
        Q[2,2] = self._Q_std**2
        Q[5,5] = self._Q_std**2

        #4. Control
        #5. Design the Measurement Function
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])
        #6. Measurement Noise Matrix
        R = np.eye(self._n_measure_vars) * self._R_std**2
        
        #7. Initial Condition
        if self._P is None:
            self._P = np.eye(self._n_state_vars) * 1.0

        return F, H, x0, self._P, Q, R#, self._n_state_vars, self._n_measure_vars


