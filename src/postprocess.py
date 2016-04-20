"""
postprocess.py
  Plot recorded IMU data (for now)

AUTHOR
  Jonathan D. Jones
"""


import os
import numpy as np

from duplocorpus import DuploCorpus
from libduplo import *


if __name__ == '__main__':
    
    corpus = DuploCorpus()
    
    devices = ('08F1', '095D', '090F', '0949')
    for trial_idx in range(corpus.meta_data.shape[0]):
        plotImuData(trial_id, devices)
        plotKinematics(trial_id, devices)
        
        corpus.makeCorrelationFigs(trial_id, devices)

    """
    #fn = '/Users/jonathan/095D.txt'
    fn = '/home/jdjones/repo/blocks/08F1_in-room_desktop.csv'
    
    data = np.loadtxt(fn, delimiter=',')
    labels = np.array([0])
    bounds = np.array([0, data.shape[0] - 1])
    
    a_norm = np.sqrt(np.sum(data[:,2:5] * data[:,2:5], 1))
    w_norm = np.sqrt(np.sum(data[:,5:8] * data[:,5:8], 1))
    b_norm = np.sqrt(np.sum(data[:,8:11] * data[:,8:11], 1))
    
    data = np.vstack((data[:,0], a_norm, w_norm, b_norm)).T
    
    txt = ('IMU sensor 2-norms', '\| \cdot \|', '??')
    plot3dof(data, labels, bounds, txt)
    plt.show()
    """