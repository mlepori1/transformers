# -*- coding: utf-8 -*-
"""
kalmanfilter.py
  Kalman filter class

AUTHOR
  Jonathan D. Jones
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt


class KalmanFilter:
    
    
    def __init__(self, init_state, init_cov):
        """
        Args:
        -----
        init_state:  (numpy vector)
          Estimated initial position in state space
        init_cov:  (numpy array)
          Covariance matrix expressing uncertainty of initial position
        """
        
        # Prediction params
        self.Xp = init_state
        self.Cp = init_cov
        
        self.num_steps_since_correction = 1
        
        #print('Initial state:  {}'.format(self.Xp.A.ravel()))
        
        # Correction params
        self.Xc = None
        self.Cc = None
    
    
    def correct(self, y, H, R):
        """
        """
        
        # FIXME: Direct matrix inversion is bad
        # FIXME: Variables must be numpy matrix objects
        K = (self.Cp * H.T) * (H * self.Cp * H.T + R).I
        
        self.Cc = self.Cp - K * H * self.Cp
        self.Xc = self.Xp + K * (y - H * self.Xp)
        
        #print('Corrected state:  {}'.format(self.Xc.A.ravel()))
        
        self.num_steps_since_correction = 0
        
        return self.Xc, self.Cc
    
    
    def predict(self, F, G, Q):
        """
        """
        
        # FIXME: predicting multiple steps into the future should be allowed
        if self.num_steps_since_correction > 0:
            self.Xp = F * self.Xp
            self.Cp = F * self.Cp * F.T + G * Q * G.T
        else:
            self.Xp = F * self.Xc
            self.Cp = F * self.Cc * F.T + G * Q * G.T
        
        #print('Predicted state:  {}'.format(self.Xp.A.ravel()))
        
        self.num_steps_since_correction += 1
        
        return self.Xp, self.Cp
    
    
    def mahalanobis(self, Ys, H, R):
        """
        """
        
        # FIXME: This only works for vector observations
        num_observations = Ys.shape[1]
        dists = np.zeros(num_observations)
        for i in range(num_observations):
            y = Ys[:,i]
            
            # Observation distribution is gaussian, so specify with mean and
            # covariance
            Yp = H * self.Xp
            cov_yp = H * self.Cp * H.T + R
            
            # Mahalanobis distance is the distance from the mean, scaled by the
            # covariance
            # FIXME: I'm not sure if this covariance matrix is necessarily positive
            #   definite
            dist_m = np.asscalar((y - Yp).T * cov_yp.I * (y - Yp)) ** 0.5
            dists[i] = dist_m
        
        return dists


if __name__ == '__main__':
    """
    Test kalman filter using simulated data
    """
    
    # Input, output, and state all have same dimension for now
    n = 2
    
    F = np.matrix(np.eye(n))
    G = np.matrix(np.eye(n))
    H = np.matrix(np.eye(n))
    
    Q = 50 * np.matrix(np.eye(n))
    R = 50 * np.matrix(np.eye(n))
    
    x0 = np.matrix(np.zeros((n,1)))
    c = np.matrix(np.eye(n))
    K = KalmanFilter(x0, c)
    
    T = 500
    u_seq = np.matrix(np.vstack((np.arange(T), np.arange(T))))
    x_seq = np.matrix(np.zeros((n, T)))
    y_seq = np.matrix(np.zeros((n, T)))
    u_err_seq = np.matrix(np.zeros((n, T)))
    x_err_seq = np.matrix(np.zeros((n, T)))
    y_err_seq = np.matrix(np.zeros((n, T)))
    x_est_seq = np.matrix(np.zeros((n, T)))
    y_est_seq = np.matrix(np.zeros((n, T)))
    for t in range(T):
        
        u = u_seq[:,t]
        x = F * x0 + G * u
        y = H * x
        
        u_err = u + Q * np.matrix(np.random.randn(n, 1))
        x_err = F * x0 + G * u_err
        y_err = H * x_err + R * np.matrix(np.random.randn(n, 1))
        
        x_est, c_est = K.correct(y_err, H, R)
        y_est = H * x_est
        x_next, c_next = K.predict(F, G, Q)
        y_next = H * x_next
        
        x_seq[:,t] = x
        y_seq[:,t] = y
        u_err_seq[:,t] = u_err
        x_err_seq[:,t] = x_err
        y_err_seq[:,t] = y_err
        x_est_seq[:,t] = x_est
        y_est_seq[:,t] = y_est
    
    f, axes = plt.subplots(3, 2)
    labels = ('u(t)', 'x(t)', 'y(t)')
    # Plot true values in red
    seqs = (u_seq, x_seq, y_seq)
    for ax, seq, label in zip(axes, seqs, labels):
        ax[0].plot(np.arange(T), seq[0,:].T.A, color='b')
        ax[1].plot(np.arange(T), seq[1,:].T.A, c='b')
        
        ax[0].set_xlabel('t')
        ax[0].set_ylabel(label)
        ax[1].set_xlabel('t')
        ax[1].set_ylabel(label)
    # Plot noisy values in blue
    seqs = (u_err_seq, x_err_seq, y_err_seq)
    for ax, seq in zip(axes, seqs):
        ax[0].plot(np.arange(T), seq[0,:].T.A, color='r')
        ax[1].plot(np.arange(T), seq[1,:].T.A, c='r')
    # Plot estimated values in green
    seqs = (x_est_seq, y_est_seq)
    for ax, seq in zip(axes[1:], seqs):
        ax[0].plot(np.arange(T), seq[0,:].T.A, color='g')
        ax[1].plot(np.arange(T), seq[1,:].T.A, c='g')
    plt.tight_layout()

        