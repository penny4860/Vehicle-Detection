# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import inv

class _BaseKF(object):

    __metaclass__ = ABCMeta
    _n_measure_vars = 2
    
    def __init__(self, n_state_vars, kf_steper, kf_designer, kf_ploter):
        """
        Parameters:
            n_state_vars (int) : number of state variables (4 or 6)
            
            kf_steper (kf_step._BaseKalmanStep) : kalman filter 1개 step을 수행하는 object

            kf_designer (kf_design._BaseKalmanDesigner) : kalman filter Design을 수행하는 object
            
            kf_ploter (kf_plot._BasePlot) : plot 을 수행하는 object

        """

        self._n_state_vars = n_state_vars
        
        self._tracker = KalmanFilter(dim_x=n_state_vars, 
                                     dim_z=self._n_measure_vars)
        
        self._kf_steper = kf_steper
        self._kf_steper.set_tracker(self._tracker)
        self._kf_designer = kf_designer
        self._kf_ploter = kf_ploter
        
        F, H, x0, P, Q, R = kf_designer.design()
        
        self._tracker.F = F
        self._tracker.Q = Q
        self._tracker.H = H
        self._tracker.R = R
        self._tracker.P = P
        self._tracker.x = x0
        
        self._init_variables()
        
    def _init_variables(self):
        self._xs = []
        self._ps = []
        self._residuals = []
        self._epsilons = []
        self._zs = None
        self._likelihoods = []

    def step(self, zs):
        self._zs = zs

        for z in zs:
            x, P, residual, eps, likelihood = self._kf_steper.epoch(z)
            
            self._xs.append(x)
            self._ps.append(P)
            self._residuals.append(residual)
            self._epsilons.append(eps)
            self._likelihoods.append(likelihood)
            
    def get_filtered_pos(self):
        stride = self._n_state_vars / self._n_measure_vars
        
        xs = np.array(self._xs)
        positions = xs[:, ::stride]
        return positions.reshape(-1, self._n_measure_vars)
    
    def plot_residual(self):
        """ 
        Description:
            Time series 별로 normalized_residual (self.epsilons) 을 plot 하는 함수. 
            self.epsilon = y.T * S * y (matrix multiplication)
        
        References:
            https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        
        """
        self._kf_ploter.plot_residual(self._epsilons)

    # TODO: truth_track 을 multiple argument로 수정해보자.
    #def plot(self, truth_track=None):
    def plot(self, track=None, label=None):
        """ measurements 와 Kalman Filtered trajectory를 plot 하는 함수. """
        measurements = self._zs
        filtered = self.get_filtered_pos()
        self._kf_ploter.plot(measurements, filtered, track, label)
        
    def plot_time_track(self, truth_track):
        """ Time series 별로 x-position, y-position 을 plot 하는 함수. """
        measurements = self._zs
        filtered = self.get_filtered_pos()
        self._kf_ploter.plot_time_track(measurements, filtered, truth_track)
    
    def evaluate(self, x_truth):
        """
        Description:
            NEES (Normalized Estimated Error Squared) 에 의해 Filter 성능 (안정성) 을 평가하는 함수.
        
        Params:
            x_truth (ndarray): Ground Truth States
            
        Return:
            epsilon (float) : evaluated error
        """    
        x_estimated = np.array(self.x).reshape(-1, self._n_state_vars)
        est_err = x_truth - x_estimated
        nees = []
        for x, p, in zip(est_err, self.P):
            nees.append(np.dot(x.T, inv(p)).dot(x))
        epsilon = np.mean(nees)
        
        print 'mean NEES is: {0}'.format(epsilon)
        if epsilon < self._n_state_vars:
            print('passed')
        else:
            print('failed')
        return epsilon
    
    def rmse(self, truth_track):
        """ Root Mean Square 에 의해 Filter 성능을 평가하는 함수. """
        filtered = self.get_filtered_pos()
        return np.sqrt(np.mean(np.sum((filtered - truth_track) ** 2, axis=1)))
            
    @property
    def x(self):
        return self._xs
    @property
    def P(self):
        return self._ps
    @property
    def epsilons(self):
        return np.array(self._epsilons).reshape(-1, )
    @property
    def residuals(self):
        return np.array(self._residuals).reshape(-1, 2)
    @property
    def tracker(self):
        return self._tracker
    @property
    def likelihoods(self):
        return self._likelihoods
    @property
    def kf_steper(self):
        return self._kf_steper
    @property
    def n_state_vars(self):
        return self._n_state_vars
    @property
    def n_measure_vars(self):
        return self._n_measure_vars

class _HybridKF(_BaseKF):
    """ 2개 이상의 Kalman Filter 를 혼합해서 사용하는 방식의 Kalman Filter """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, kf_steper, kf_ploter):
        """
        Parameters:
            kf_bank (list of kf_base._BaseKF)
            
            probs (list)
                Model 별 initial probability
                
            transtion_probs (list of list)
                Model 사이의 transition probability
                
            model_names (list of string)
                Model 의 이름
        """
        self._kf_steper = kf_steper
        self._kf_designer = None
        self._kf_ploter = kf_ploter

        self._init_variables()
        self._model_probs_history = []
        
    def _init_variables(self):
        self._xs = []
        self._ps = []
        self._epsilons = []
        self._zs = None
        self._model_probs_history = []

    def step(self, zs):
        self._zs = zs

        for z in zs:
            x, P, _, eps, _, model_probs = self._kf_steper.epoch(z)
            
            self._xs.append(x)
            self._ps.append(P)
            self._epsilons.append(eps)
            self._model_probs_history.append(model_probs)

    def plot_model_prob(self):
        self._kf_ploter.plot_model_prob(self.model_probs_history)
    
    @property
    def model_probs_history(self):
        return np.array(self._model_probs_history)
        
        
        
