# -*- coding: utf-8 -*-
from math import sqrt
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

FIRMWARE_PORTING_LOG_EACH_MODEL = False
FIRMWARE_PORTING_LOG_KF_INIT = False
FIRMWARE_PORTING_LOG_KF_SUM = False

def interpolate(i, index1, index2, value1, value2, boost):
    
    value = int(abs(i - index1) * value2 + abs(i - index2) * value1) / boost
    return value

def exp_approximate(i):
    """ 
    exp(i) 연산을 근사하는 함수.
    """
    EXP_MINUS_TABLE = [
        10000,                # exp(0) = 1.0
         3678,     # exp(-1) = 0.367879441171
         1353, 
          497, 
          183, 
           67, 
           24,   # exp(-6) = 0.00247875217667
            0                 # exp(-7) = 0.0 으로 간주
    ]

    assert i <= 0, "i must be less than 0"
    
    boost = 10000
    _i = int(-i * boost)
    index_i = _i / boost
    
    if index_i >= len(EXP_MINUS_TABLE)-1:
        return 0
    else:
        index1 = index_i + 1
        index2 = index_i
        value1 = EXP_MINUS_TABLE[index1]
        value2 = EXP_MINUS_TABLE[index2]
        exp_value = interpolate(_i, index1*boost, index2*boost, value1, value2, boost)
    
        return int(exp_value)
        
class _BaseKalmanStep(object):
    """ Kalman Filter의 1-step Filtering algorithm 을 담당하는 abstract class """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    @abstractmethod
    def epoch(self, z):
        pass
    
class _SingleModelKalmanStep(_BaseKalmanStep):
    """ Single Model을 사용하는 Kalman Filter의 1-step Filtering algorithm 을 담당하는 abstract class """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
        
    def set_tracker(self, tracker):
        self._tracker = tracker
        
    def epoch(self, z, unit_boosting=10):
        
        # 1. Prediction : self._tracker.predict()
        self._tracker.x = np.dot(self._tracker.F, self._tracker.x)
        self._tracker.P = np.dot(self._tracker.F, self._tracker.P).dot(self._tracker.F.T) + self._tracker.Q
        
        if FIRMWARE_PORTING_LOG_EACH_MODEL:
            print
            print "> [166] Frame Start--------------------------------------------"
            print "[246] Measurement z"
            print z
            print "[213] P (Prediction Covariance)"
            print self._tracker.P
            print "[220] Predict State (X(k|k-1))"
            print self._tracker.x.T
        
        z = z.reshape(-1,1)
        # 2. Prediction : self._tracker.update(z.reshape(-1,1))
        self._tracker._y = z - np.dot(self._tracker.H, self._tracker.x)
        self._tracker._y = self._tracker._y.astype(int)


        self._tracker.P = self._tracker.P.astype(np.float32)
        self._tracker._S[0,0] = int(self._tracker.P[0, 0]) + int(self._tracker.R[0, 0])
        self._tracker._S[1,1] = int(self._tracker.P[0, 0]) + int(self._tracker.R[1, 1])
        self._tracker._S[1,0] = 0
        self._tracker._S[0,1] = 0
        
        K_BOOSTING = 1000
        self._tracker._K = np.zeros((6, 2))
        self._tracker._K[0, 0] = int(self._tracker.P[0, 0] * K_BOOSTING / self._tracker._S[0,0])
        self._tracker._K[1, 0] = int(self._tracker.P[1, 0] * K_BOOSTING / self._tracker._S[0,0])
        self._tracker._K[2, 0] = int(self._tracker.P[2, 0] * K_BOOSTING / self._tracker._S[0,0])
        self._tracker._K[3, 1] = int(self._tracker.P[0, 0] * K_BOOSTING / self._tracker._S[0,0])
        self._tracker._K[4, 1] = int(self._tracker.P[1, 0] * K_BOOSTING / self._tracker._S[0,0])
        self._tracker._K[5, 1] = int(self._tracker.P[2, 0] * K_BOOSTING / self._tracker._S[0,0])

        self._tracker._x = self._tracker.x + np.dot(self._tracker.K, self._tracker._y) / K_BOOSTING
        self._tracker._P = (np.identity(6) - np.dot(self._tracker.K, self._tracker.H) / K_BOOSTING).dot(self._tracker.P)

        if FIRMWARE_PORTING_LOG_EACH_MODEL:
            print "[216] K (Kalman Gain)"
            print self._tracker.K
            
            print "[218] Posterior State (X(k|k))"
            print self._tracker.x.T
    
            print "[219] P(k|k): Posterior Covariance"
            print self._tracker.P
        
        y = self._tracker.y
        S = self._tracker.S
        if FIRMWARE_PORTING_LOG_EACH_MODEL:
            print "[215] S (residual Covariance)"
            print S
            print "[222] Y (Residual)"
            print y.T
                    
        eps = np.dot(np.dot(y.T, inv(S)), y)/(unit_boosting**2)
        eps = eps[0, 0]
        if FIRMWARE_PORTING_LOG_EACH_MODEL:
            print "[245] eps, (eps*10000)"
            print eps, eps*10000
        
        #likelihood = exp_approximate(-eps/2) / np.sqrt(np.linalg.det(S))
        # exp_approximate(-eps/2) 의 최대값은 10000
        # 10000 * 10000 == 10 ** 8 < 2**32 == 2 ** 10 * 4 == 4 * 10 ** 9 
        likelihood = int(exp_approximate(-eps/2) * 1000) / int(S[0, 0])
        
        if FIRMWARE_PORTING_LOG_EACH_MODEL:
            print "[248] likelihood"
            print likelihood
        
        #self._tracker.P = (self._tracker.P).astype(int)
        
#         self._tracker.P = (self._tracker.P * 1000).astype(int)
#         self._tracker.P = self._tracker.P.astype(float) / 1000
                
        return self._tracker.x, self._tracker.P, self._tracker.y, eps, likelihood
    
    @property
    def tracker(self):
        return self._tracker

class BasicKalmanStep(_SingleModelKalmanStep):
    pass

class FadingMemoryKalmanStep(_SingleModelKalmanStep):

    def __init__(self, alpha):
        self._alpha = alpha
        
    def set_tracker(self, tracker):
        self._tracker = tracker
        self._tracker.alpha = self._alpha
        
class AdaptiveEpsKalmanStep(_SingleModelKalmanStep):
    
    def __init__(self, Q_scale_factor=1000., eps_max=4.):
        self._Q_scale_factor = Q_scale_factor
        self._eps_max = eps_max
        
        self._count = 0

    def epoch(self, z):
        
        _, _, _, eps, _ = super(AdaptiveEpsKalmanStep, self).epoch(z)
        
        if eps > self._eps_max:
            self._tracker.Q *= self._Q_scale_factor
            self._count += 1
        elif self._count > 0:
            self._tracker.Q /= self._Q_scale_factor
            self._count -= 1
            
        likelihood = self._tracker.likelihood
        
        return self._tracker.x, self._tracker.P, self._tracker.y, eps, likelihood

class AdaptiveStdKalmanStep(_SingleModelKalmanStep):
    
    def __init__(self, Q_std, Q_scale_factor=1000., std_scale=4.):
        self._Q_scale_factor = Q_scale_factor
        self._std_scale = std_scale
        
        self._count = 0
        self._phi = Q_std

    def epoch(self, z):
        
        dt = 1.0
        _, _, _, eps, _ = super(AdaptiveStdKalmanStep, self).epoch(z)
        
        mean_var = np.mean(np.diag(self._tracker.S))
        mean_residual = np.mean(self._tracker.y)
         
        std = sqrt(mean_var)
        dim = len(self._tracker.Q) / 2

        if abs(mean_residual) > self._std_scale*std:
            self._phi += self._Q_scale_factor
            q = Q_discrete_white_noise(dim=dim, dt=dt, var=self._phi**2)
            self._tracker.Q = block_diag(q, q)
            self._count += 1
            
        elif self._count > 0:
            self._phi -= self._Q_scale_factor
            q = Q_discrete_white_noise(dim=dim, dt=dt, var=self._phi**2)
            self._tracker.Q = block_diag(q, q)
            self._count -= 1

        likelihood = self._tracker.likelihood
        
        return self._tracker.x, self._tracker.P, self._tracker.y, eps, likelihood


class _MultipleModelKalmanStep(_BaseKalmanStep):
    """ Multiple Model을 사용하는 Kalman Filter의 1-step Filtering algorithm 을 담당하는 abstract class """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, stepers, probs, transition_probs, unit_boosting=1):
        """
        Parameters:
            unit_boosting (int) : 
                Input 좌표의 boosting 되어있는 정도. 
                e.g.) unit_boosting=10 이라면 (2160.0, 2360.0)는 (216, 236) 으로 간주하고 Kalman Filter 의 IMM 이 동작한다.
                position-domain 의 차이는 epsilon 과 likelihood 계산에 영향을 미친다. 
        """
        self._stepers = stepers
        self._probs = np.array(probs)
        self._trans = np.array(transition_probs)
        
        self._unit_boosting = unit_boosting

    def epoch(self, z):
    
        # filter 별로 epoch 의 결과를 저장하기 위한 buffer list
        xs = []
        Ps = []
        ys = []
        epss = []
        likelihoods = []
        M = self._trans
        
        # filter bank 별로 epoch를 수행
        for steper in self._stepers:
            x, P, y, eps, likelihood = steper.epoch(z, unit_boosting=self._unit_boosting)
            
            x = np.array(x)
                            
            xs.append(x)
            Ps.append(P)
            ys.append(y)
            epss.append(eps)
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        
        prior_probs = np.dot(self._probs, M)
        posteriors = np.array([likelihood * prior for likelihood, prior in zip(likelihoods, prior_probs)])
        posteriors /= np.sum(posteriors)

        self._probs = posteriors
        epss = np.array(epss).reshape(-1,)

        x = np.zeros_like(xs[0])
        P = np.zeros_like(Ps[0])
        for posterior_, x_, p_ in zip(posteriors, xs, Ps):
            #print P_.shape, y_.shape, eps_.shape
            x += x_ * posterior_
            P += p_ * posterior_

        if FIRMWARE_PORTING_LOG_KF_SUM:
            print
            print "> [166] Frame Start--------------------------------------------"
            print "[249] Prior Probability * 10000"
            print prior_probs * 10000
            print "[248] likelihood"
            print likelihoods
            print "[230] Posterior (*10000)"
            print posteriors * 10000
            print "[232] Filtered Pos X, Y"
            print x[0], x[3]
            
        return x, P, ys, epss, likelihoods, posteriors


class MmaeKalmanStep(_MultipleModelKalmanStep):
    """
    Description: 
        Multiple Model Adaptive Estimator (MMAE) 의 1-step Filtering algorithm 을 담당하는 class
        
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        
        http://www.eecs.berkeley.edu/~tomlin/papers/journals/hbt06_iee.pdf
            ("State estimation for hybrid systems: applications to aircraft tracking")
        
    """
    
    # _MultipleModelKalmanStep class 의 구현을 상속받아 그대로 사용한다.
    pass


class ImmKalmanStep(_MultipleModelKalmanStep):
    """
    Description: 
        Interacting Multiple Models (IMM) 의 1-step Filtering algorithm 을 담당하는 class
        
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        
        http://www.eecs.berkeley.edu/~tomlin/papers/journals/hbt06_iee.pdf
            ("State estimation for hybrid systems: applications to aircraft tracking")
            
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.9763&rep=rep1&type=pdf
            ("A comparative study of multiple-model algorithms for maneuvering target tracking")
        
    """
    def epoch(self, z):

        N = len(self._stepers)
        
        # 1. Mixing Probabilities
        mu_k = self._probs
        M = self._trans
        
        # normalizer = np.dot(self._probs, self._trans)
        cbar = np.dot(mu_k, M)
        
        if FIRMWARE_PORTING_LOG_KF_INIT:
            print 
            print "===================================Frame Start============================================="
            print "  [249] Prior Probability (cbar)"
            print "    ", cbar
            print "    ", cbar * 10000
        
        mu_k_k = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                mu_k_k[i, j] = M[i, j] * mu_k[i] / cbar[j]

        if FIRMWARE_PORTING_LOG_KF_INIT:
            print "  [226] Mixing Probablity x10000 (mu_k_k)"
            for m in mu_k_k:
                print "    ", m * 10000
        
        # 2. New Initial State, Covariance
        init_xs = []
        init_ps = []
        
        for j in range(N):
            state_sum = 0
            for i in range(N):
                state_sum += self._stepers[i].tracker.x * mu_k_k[i, j]
            init_xs.append(state_sum)
        init_xs = np.array(init_xs)        

        if FIRMWARE_PORTING_LOG_KF_INIT:
            print "  [228] New Initial State X0 (* 1000)"
            for x in init_xs:
                print "    ", x.T * 1000

        for j in range(N):
            cov_sum = 0
            for i in range(N):
                P = self._stepers[i].tracker.P
                x = self._stepers[i].tracker.x
                #print P.shape, x.shape, init_xs[j].shape
                #cov_sum += (P + np.dot((x - init_xs[j]), (x - init_xs[j]).T)  )* mu_k_k[i, j]
                cov_sum += (P)* mu_k_k[i, j]
            init_ps.append(cov_sum)
        init_ps = np.array(init_ps) # (N, 6, 6)

        if FIRMWARE_PORTING_LOG_KF_INIT:
            print "  [227] New Initial P0 (* 100)"
            for ps in init_ps:
                for p in ps:
                    print "    ", p * 100
                print

        # Update steper's state & covariance to init_x, init_p
        for i in range(N):
            self._stepers[i].tracker.x = init_xs[i]
            self._stepers[i].tracker.P = init_ps[i]
        
        return super(ImmKalmanStep, self).epoch(z)

if __name__ == "__main__":
    
    exp_approximate(-0.040517374911)
    
    
    
    