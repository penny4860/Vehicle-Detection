# -*- coding: utf-8 -*-

import data_generator
import kf_step
import kf_design
import kf_plot
import kf_base

class ConstTurnKF(kf_base._BaseKF):
    """ Constant Turn Model 을 사용하는 Kalman Filter """
    
    # state variable은 6개 (pos_x, vel_x, acc_x, pos_y, vel_y, acc_y)
    # Original 등속도 모델은 4개의 state 만 사용하지만 Hybrid Model 에서 
    # Combine 을 쉽게 하기 위해 6개의 state를 사용하였다. 
    _n_state_vars = 6
    
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, dt=1.0, w=3):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """

        kf_designer = kf_design.ConstTurnDesigner(init_pos, P, Q_std, R_std, dt=dt, w=w)
        kf_steper = kf_step.BasicKalmanStep()
        kf_ploter = kf_plot.SingleModelPlot()

        super(ConstTurnKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)


class ConstVelKF(kf_base._BaseKF):
    """ Constant Velocity Model 을 사용하는 Kalman Filter """
    
    # state variable은 6개 (pos_x, vel_x, acc_x, pos_y, vel_y, acc_y)
    # Original 등속도 모델은 4개의 state 만 사용하지만 Hybrid Model 에서 
    # Combine 을 쉽게 하기 위해 6개의 state를 사용하였다. 
    _n_state_vars = 6
    
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """

        kf_designer = kf_design.ConstVelDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.BasicKalmanStep()
        kf_ploter = kf_plot.SingleModelPlot()

        super(ConstVelKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)

        
class StationKF(kf_base._BaseKF):
    """ Stationary Model 을 사용하는 Kalman Filter """
    
    # state variable은 6개 (pos_x, vel_x, acc_x, pos_y, vel_y, acc_y)
    # Original 등속도 모델은 4개의 state 만 사용하지만 Hybrid Model 에서 
    # Combine 을 쉽게 하기 위해 6개의 state를 사용하였다. 
    _n_state_vars = 6
     
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """

        
        kf_designer = kf_design.StationaryDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.BasicKalmanStep()
        kf_ploter = kf_plot.SingleModelPlot()

        super(StationKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)
        
    
class ConstAccKF(kf_base._BaseKF):
    """ Constant Acceleration Model 을 사용하는 Kalman Filter """
     
    # state variable은 6개 (pos_x, vel_x, acc_x, pos_y, vel_y, acc_y)
    _n_state_vars = 6
     
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """
 
        kf_designer = kf_design.ConstAccDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.BasicKalmanStep()
        kf_ploter = kf_plot.SingleModelPlot()

        super(ConstAccKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)

class AdaptiveEpsKF(kf_base._BaseKF):
    """
    Description:
        Constant Velocity를 Prediction Model로 사용하고 normalized residual (epsillon) 에 따라서 
        Q_val 를 adaptive 하게 변화시키는 Kalman Filter 
    
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        [Continuous Adjustment]
    """
    _n_state_vars = 6
     
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, Q_scale_factor=1000., eps_max=4., dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            Q_scale_factor (float):
                normalized residual (epsilon) 이 커졌을 때 Q_std에 더해지는 Scale Factor.
                setting value 가 클수록 변화에 빨리 반응하게 된다.
                
            eps_max (float)
                Q 값을 증가하는 기준을 정하는 수식에 관계된 Parameter.
                setting value 가 작을 수록 변화에 빨리 반응하게 된다.
            
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """

 
        kf_designer = kf_design.ConstVelDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.AdaptiveEpsKalmanStep(Q_scale_factor, eps_max)
        kf_ploter = kf_plot.SingleModelPlot()

        super(AdaptiveEpsKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)
        
class AdaptiveStdKF(kf_base._BaseKF):
    """
    Description:
        Constant Velocity를 Prediction Model로 사용하고 std 에 따라서 Q_val 를 adaptive 하게 변화시키는 Kalman Filter 
    
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        [Continuous Adjustment - Standard Deviation Version]

    """
    _n_state_vars = 6
     
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, Q_scale_factor=1000., std_scale=4., dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            Q_scale_factor (float):
                standard deviation 이 커졌을 때 Q_std에 더해지는 Scale Factor.
                setting value 가 클수록 변화에 빨리 반응하게 된다.
                
            std_scale (float)
                Q 값을 증가하는 기준을 정하는 수식에 관계된 Parameter.
                setting value 가 작을 수록 변화에 빨리 반응하게 된다.
            
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """

 
        kf_designer = kf_design.ConstVelDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.AdaptiveStdKalmanStep(Q_std, Q_scale_factor, std_scale)
        kf_ploter = kf_plot.SingleModelPlot()

        super(AdaptiveStdKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)


class FadingMemoryKF(kf_base._BaseKF):
    """
    Description:
        current measurement 에 weighting 을 줘서 maneuvering object 의 lagging을 최소화 하는 Kalman Filter
    
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        [Fading Memory Filter]

    """
    _n_state_vars = 6
     
    def __init__(self, init_pos, P=None, Q_std=1.0, R_std=1.0, alpha=1.05, dt=1.0):
        """
        Parameters:
            init_pos (tuple):
                Kalman Filter 에서 Initial State 에 사용할 position 정보.
                Measurement 의 첫 번째 값을 그대로 사용하도록 하자.
                
            P (ndarray):
                Initial Covariance Matrix
                
            Q_std (float):
                Prediction Error 에 대한 Standard Variation.
                setting한 scalar 값에 따라 Q matrix 가 자동으로 계산된다.
                
            R_std (float)
                Measurement Error 에 대한 Standard Variation.
        
            alpha (float): 
                Fading Memory Factor. value 가 클수록 현재의 measurement 에 weighting 이 커진다.
                1.0 보다 큰 값을 쓰도록 하자.
            
            dt (float):
                time-step. time step을 0.1 정도로 작게 설정하는 것이 Filter 동작이 안정적이다.
                time-step 을 1로 사용하면 Q값을 10**-6 정도로 작게 써야한다.
        """
 
        kf_designer = kf_design.ConstVelDesigner(init_pos, P, Q_std, R_std, dt=dt)
        kf_steper = kf_step.FadingMemoryKalmanStep(alpha)
        kf_ploter = kf_plot.SingleModelPlot()

        super(FadingMemoryKF, self).__init__(self._n_state_vars, kf_steper, kf_designer, kf_ploter)


class MMAE(kf_base._HybridKF):
    """
    Description:
        2개 이상의 Kalman Filter Model 을 사용하고, Filtering 결과를 Bayesian Rule 에 따라 Interpolation 하는 
        Hybrid Model Kalman Filter 를 구현한 class.
        Filter 의 상세한 알고리즘은 MmaeKalmanStep class 를 참조하자.
    
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb [MMAE]
        
        http://www.eecs.berkeley.edu/~tomlin/papers/journals/hbt06_iee.pdf
            ("State estimation for hybrid systems: applications to aircraft tracking")
    """
    _n_state_vars = 6
     
    def __init__(self, kf_bank, probs, transtion_probs, model_names=None):
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
        stepers = [kf.kf_steper for kf in kf_bank]
        kf_steper = kf_step.MmaeKalmanStep(stepers, probs, transtion_probs)
        kf_ploter = kf_plot.MultipleModelPlot(len(kf_bank), model_names)

        super(MMAE, self).__init__(kf_steper, kf_ploter)
        
class IMM(kf_base._HybridKF):
    """
    Description:
        2개 이상의 Kalman Filter Model 을 사용하고, Filtering 결과를 Bayesian Rule 에 따라 Interpolation 하는 
        Hybrid Model Kalman Filter 를 구현한 class.
        
        MMAE 와 다르게 Kalman Filter 각각이 독립적으로 Filtering 을 수행하기 전에 Initial State, 
        Initial Covariance 를 수정하는 과정을 수행한다. 
        Filter 의 상세한 알고리즘은 ImmKalmanStep class 를 참조하자.
    
    References:
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb [IMM]
        
        http://www.eecs.berkeley.edu/~tomlin/papers/journals/hbt06_iee.pdf
            ("State estimation for hybrid systems: applications to aircraft tracking")
            
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.9763&rep=rep1&type=pdf
            ("A comparative study of multiple-model algorithms for maneuvering target tracking")
    """
    _n_state_vars = 6
     
    def __init__(self, kf_bank, probs, transtion_probs, model_names=None, unit_boosting=1):
        """
        Parameters:
            kf_bank (list of kf_base._BaseKF)
            
            probs (list)
                Model 별 initial probability
                
            transtion_probs (list of list)
                Model 사이의 transition probability

            model_names (list of string)
                Model 의 이름
                
            unit_boosting (int) : 
                Input 좌표의 boosting 되어있는 정도. 
                e.g.) unit_boosting=10 이라면 (2160.0, 2360.0)는 (216, 236) 으로 간주하고 Kalman Filter 의 IMM 이 동작한다.
                position-domain 의 차이는 epsilon 과 likelihood 계산에 영향을 미친다. 

        """
        stepers = [kf.kf_steper for kf in kf_bank]
        kf_steper = kf_step.ImmKalmanStep(stepers, probs, transtion_probs, unit_boosting=unit_boosting)
        kf_ploter = kf_plot.MultipleModelPlot(len(kf_bank), model_names)

        super(IMM, self).__init__(kf_steper, kf_ploter)

        
if __name__ == "__main__":
    
    # Generate Parameters
    ###################################
    sensor_std = 5.0
    ###################################
    
    # 1. Track Generation
    #tr = data_generator.LineTrack()
    #tr = data_generator.CurveTrack()
    #tr = data_generator.CircleTrack()
    #tr = data_generator.StationaryTrack()
    #tr = data_generator.TrackFromFile("..//tests//trajectory2.txt")
    tr = data_generator.TrackFromFile("..//tests//trajectory6.txt")

    
    tr.generate_track()
    tr.generate_zs(std_noise=sensor_std, seed=111)
    zs = tr.get_measured_pos()

    # 2. Kalman Filter
    import numpy as np
    Q_value = 1.0
    R_value = 100.0
    P = np.eye(6) * 1000
    
    cv = ConstVelKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P)
    ca = ConstAccKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P)
    ct0 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=1.0)
    ct1 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=-1.0)
    ct2 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=0.5)
    ct3 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=-0.5)
    ct4 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=0.1)
    ct5 = ConstTurnKF((zs[0, 0], zs[0, 1]), Q_std=Q_value, R_std=R_value, P=P, w=-0.1)

    kf_bank = [cv, ca, ct0, ct1, ct2, ct3, ct4, ct5]
    #kf_bank = [cv, ca, ct0, ct1, ct2, ct3]
    #kf_bank = [cv, ca, ct4, ct5]
    
    model_names=["CV", "CA", "CT +1", "CT -1", "CT +0.5", "CT -0.5", "CT +1", "CT -1"]
    #model_names=["CV", "CA", "CT +1", "CT -1"]
    
    n_filters = len(kf_bank)
    probs = np.ones((n_filters)) / n_filters
    transtion_probs = np.zeros((n_filters, n_filters))
    
    KEEP = 0.8
    for i in range(n_filters):
        for j in range(n_filters):
            if i == j:
                transtion_probs[i, j] = KEEP
            else:
                transtion_probs[i, j]  = (1.0 - KEEP) / (n_filters-1)
    
    kf = IMM(kf_bank = kf_bank, 
             probs = probs, 
             transtion_probs = transtion_probs,
             model_names=model_names
             )
    
    kf.step(zs)
    kf.evaluate(tr.get_truth_state(n_states=6))
    print kf.rmse(tr.get_truth_pos())
    
    # 3. Plot
    #kf.plot_residual()
#     kf.plot(tr.get_truth_pos())
#     kf.plot_model_prob()
    #kf.plot_time_track(tr.get_truth_pos())

    
    
    


