"""
postprocess.py
  Plot recorded IMU data (for now)

AUTHOR
  Jonathan D. Jones
"""


import os
import numpy as np
from matplotlib import pyplot as plt


def plotImuData(t_init, devices, sample_len):
    """
    Create plots of IMU acceleration, angular velocity, magnetic field
    
    Args:
    -----
      t_init:
      devices:
      sample_len:
    """

    fn = os.path.join('data', 'imu', '{}.csv'.format(t_init))
    data = np.loadtxt(fn, delimiter=',')
    num_samples, num_vars = data.shape
    assert(num_vars % sample_len == 0)
    
    imus = []
    for i in range(int(num_vars / sample_len)):
        i_start = i * sample_len
        i_end = (i + 1) * sample_len
        imus.append(data[:,i_start:i_end]) 
    
    # Plot accelerometer, gyroscope, magnetometer readings
    fa, ax_a = plt.subplots(3, 1)
    fg, ax_g = plt.subplots(3, 1)
    fm, ax_m = plt.subplots(3, 1)
    
    for i, imu in enumerate(imus):
    
        # Filter out bad data for now b/c it makes the plots impossible to read
        bad_data = np.less(imu[:,0], 1.0)
        imu = imu[np.logical_not(bad_data),:]
        name = devices[i]
    
        # Calculate time relative to trial start
        imu[:,0] = imu[:,0] - t_init
    
        # Accelerometer range setting +/- 8g --> divide by 4096 to get units of g
        imu[:,2:5] = imu[:,2:5] / 4096.0
        ax_a[0].plot(imu[:,0], imu[:,2], label=name)
        ax_a[1].plot(imu[:,0], imu[:,3], label=name)
        ax_a[2].plot(imu[:,0], imu[:,4], label=name)
    
        # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get units
        # of degrees per second
        imu[:,5:8] = imu[:,5:8] * 0.07 / 180
        ax_g[0].plot(imu[:,0], imu[:,5], label=name)
        ax_g[1].plot(imu[:,0], imu[:,6], label=name)
        ax_g[2].plot(imu[:,0], imu[:,7], label=name)
    
        # Multiply by 1e-4 to get units of mT, but this will be very approximate
        # (see WAX9 developer guide)
        imu[:,8:] = imu[:,8:] * 1e-4
        ax_m[0].plot(imu[:,0], imu[:,8], label=name)
        ax_m[1].plot(imu[:,0], imu[:,9], label=name)
        ax_m[2].plot(imu[:,0], imu[:,10], label=name)
    
    for axes in (ax_a, ax_g, ax_m):
        for ax in axes:
            ax.set_xlabel('t (s)')
    
    ax_a[0].set_title('Acceleration in IMU frame')
    ax_a[0].set_ylabel(r'$a_x$ (g)')
    ax_a[1].set_ylabel(r'$a_y$ (g)')
    ax_a[2].set_ylabel(r'$a_z$ (g)')
    
    ax_g[0].set_title('Angular velocity in IMU frame')
    ax_g[0].set_ylabel(r'$\omega_x$ ($\pi$ rad/s)')
    ax_g[1].set_ylabel(r'$\omega_y$ ($\pi$ rad/s)')
    ax_g[2].set_ylabel(r'$\omega_z$ ($\pi$ rad/s)')
    
    ax_m[0].set_title('Magnetic field in IMU frame')
    ax_m[0].set_ylabel(r'$B_x$ (mT)')
    ax_m[1].set_ylabel(r'$B_y$ (mT)')
    ax_m[2].set_ylabel(r'$B_z$ (mT)')
    
    for fig in (fa, fg, fm):
        fig.tight_layout()
    
    for axes in (ax_a, ax_g, ax_m):
        for ax in axes:
            # Shrink plot width by 20% to make room for the legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


if __name__ == '__main__':
    t_init = 1457450852
    devices = ('08F1', '095D', '090F', '0949')
    plotImuData(t_init, devices, 11)