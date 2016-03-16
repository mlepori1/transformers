"""
postprocess.py
  Plot recorded IMU data (for now)

AUTHOR
  Jonathan D. Jones
"""


import os
import numpy as np
from matplotlib import pyplot as plt


def convertFrameIndex(video_timestamps, kinem_timestamps):
    """
    Find the indices of the kinematic timestamps that most closely
    match the video timestamps.
    
    Args:
    -----
    [np vector] video_timestamps: The i-th entry contains the universal time
      at which the i-th VIDEO FRAME occurred
    [np vector] kinem_timestamps: The i-th entry contains the universal time
      at which the i-th KINEMATIC SAMPLE occurred
      
    Returns:
    --------
    [np vector] indices: Same length as video_timestamps.
      The i-th entry holds the index of kinem_timestamps that most
      closely matches the time of the i-th index in video_timestamps.
    """
    
    vid_len = video_timestamps.shape[0]
    kin_len = kinem_timestamps.shape[0]
            
    # For each video timestamp, find the index of the nearest
    # kinematic timestamp (+/- one frame)
    # Video and kinematic timestamps are in increasing order.
    indices = np.zeros((vid_len,), dtype=int)
    j = 0
    for i in range(vid_len):
        while video_timestamps[i] > kinem_timestamps[j] \
          and j < kin_len - 1:
            j += 1
        indices[i] = j
        
    return indices


def plotImuData(t_init, devices, sample_len):
    """
    Create plots of IMU acceleration, angular velocity, magnetic field
    
    Args:
    -----
      [int] t_init:
      [list(str)] devices:
      [int] sample_len:
    """

    # Load IMU data
    fn = os.path.join('data', 'imu', '{}.csv'.format(t_init))
    data = np.loadtxt(fn, delimiter=',')
    num_samples, num_vars = data.shape
    assert(num_vars % sample_len == 0)
    
    imus = []
    for i in range(int(num_vars / sample_len)):
        i_start = i * sample_len
        i_end = (i + 1) * sample_len
        imus.append(data[:,i_start:i_end])
    
    # Load RGB frame timestamps
    fn = os.path.join('data', 'rgb', str(t_init), 'frame-timestamps.csv')
    video_timestamps = np.loadtxt(fn)
    
    # Load labels if we have them (file contains indices and descriptions)
    fn = os.path.join('data', 'labels', '{}.csv'.format(t_init))
    if os.path.exists(fn):
        typestruct = [('index', 'i'), ('comment', 'S')]
        labels = np.loadtxt(fn, delimiter=',', dtype=typestruct)
        labels = labels['index']
    else:
        print('Trial {}: no labels found'.format(t_init))
        labels = []
    
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    # Plot accelerometer, gyroscope, magnetometer readings
    selected_devices = range(len(imus))
    for i in selected_devices:
        
        imu = imus[i]
        name = devices[i]
        
        # Convert video frame labels to IMU indices
        vid_idxs = convertFrameIndex(video_timestamps, imu[:,0])
        label_idxs = vid_idxs[labels]
    
        # Filter out bad data for now b/c it makes the plots impossible to read
        bad_data = np.less(imu[:,0], 1.0)
        imu = imu[np.logical_not(bad_data),:]
    
        # Calculate time relative to trial start
        imu[:,0] = imu[:,0] - t_init
    
        # Accelerometer range setting +/- 8g --> divide by 4096 to get units of g
        imu[:,2:5] = imu[:,2:5] / 4096.0
        fa, ax_a = plt.subplots(3, 1)
        ax_a[0].plot(imu[:,0], imu[:,2], label=name)
        ax_a[1].plot(imu[:,0], imu[:,3], label=name)
        ax_a[2].plot(imu[:,0], imu[:,4], label=name)
        ax_a[0].scatter(imu[label_idxs,0], imu[label_idxs,2], c='k')
        ax_a[1].scatter(imu[label_idxs,0], imu[label_idxs,3], c='k')
        ax_a[2].scatter(imu[label_idxs,0], imu[label_idxs,4], c='k')
        ax_a[0].set_title('Acceleration in IMU frame')
        ax_a[0].set_ylabel(r'$a_x$ (g)')
        ax_a[1].set_ylabel(r'$a_y$ (g)')
        ax_a[2].set_ylabel(r'$a_z$ (g)')
    
        # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get units
        # of degrees per second
        imu[:,5:8] = imu[:,5:8] * 0.07 / 180
        fg, ax_g = plt.subplots(3, 1)
        ax_g[0].plot(imu[:,0], imu[:,5], label=name)
        ax_g[1].plot(imu[:,0], imu[:,6], label=name)
        ax_g[2].plot(imu[:,0], imu[:,7], label=name)
        ax_g[0].scatter(imu[label_idxs,0], imu[label_idxs,5], c='k')
        ax_g[1].scatter(imu[label_idxs,0], imu[label_idxs,6], c='k')
        ax_g[2].scatter(imu[label_idxs,0], imu[label_idxs,7], c='k')
        ax_g[0].set_title('Angular velocity in IMU frame')
        ax_g[0].set_ylabel(r'$\omega_x$ ($\pi$ rad/s)')
        ax_g[1].set_ylabel(r'$\omega_y$ ($\pi$ rad/s)')
        ax_g[2].set_ylabel(r'$\omega_z$ ($\pi$ rad/s)')
    
        # Multiply by 1e-4 to get units of mT, but this will be very approximate
        # (see WAX9 developer guide)
        imu[:,8:] = imu[:,8:] * 1e-4
        fm, ax_m = plt.subplots(3, 1)
        ax_m[0].plot(imu[:,0], imu[:,8], label=name)
        ax_m[1].plot(imu[:,0], imu[:,9], label=name)
        ax_m[2].plot(imu[:,0], imu[:,10], label=name)
        ax_m[0].scatter(imu[label_idxs,0], imu[label_idxs,8], c='k')
        ax_m[1].scatter(imu[label_idxs,0], imu[label_idxs,9], c='k')
        ax_m[2].scatter(imu[label_idxs,0], imu[label_idxs,10], c='k')
        ax_m[0].set_title('Magnetic field in IMU frame')
        ax_m[0].set_ylabel(r'$B_x$ (mT)')
        ax_m[1].set_ylabel(r'$B_y$ (mT)')
        ax_m[2].set_ylabel(r'$B_z$ (mT)')
    
        # x label is the same for all axes
        for axes in (ax_a, ax_g, ax_m):
            for ax in axes:
                ax.set_xlabel('t (s)')
        
        # Set tight layout BEFORE shrinking plot otherwise figures are wonky
        for fig in (fa, fg, fm):
            fig.tight_layout()
        
        # Shrink plot width by 20% to make room for the legend
        for axes in (ax_a, ax_g, ax_m):
            for ax in axes:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save plots to IMU figures directory and DON'T show them
        for fig, reading in zip((fa, fg, fm), ('accel', 'ang-vel', 'mag')):
            fname = '{}_{}.png'.format(reading, name)
            fig.savefig(os.path.join(imu_fig_path, fname))
            plt.close()
    
    #plt.show()


if __name__ == '__main__':
    t_init = 1457987953
    devices = ('08F1', '095D', '090F', '0949')
    plotImuData(t_init, devices, 11)