"""
postprocess.py
  Plot recorded IMU data (for now)

AUTHOR
  Jonathan D. Jones
"""


import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def vidFrame2imuFrame(vid_timestamps, imu_timestamps):
    """
    Find the indices of the IMU timestamps that most closely
    match the video timestamps.
    
    Args:
    -----
    [np vector] vid_timestamps: The i-th entry contains the universal time
      at which the i-th VIDEO FRAME occurred
    [np vector] imu_timestamps: The i-th entry contains the universal time
      at which the i-th IMU SAMPLE occurred
      
    Returns:
    --------
    [np vector] indices: Same length as vid_timestamps.
      The i-th entry holds the index of imu_timestamps that most
      closely matches the time of the i-th index in video_timestamps.
    """
    
    vid_len = vid_timestamps.shape[0]
    imu_len = imu_timestamps.shape[0]
            
    # For each video timestamp, find the index of the nearest
    # IMU timestamp (+/- one frame)
    # Video and IMU timestamps are in increasing order.
    indices = np.zeros((vid_len,), dtype=int)
    j = 0
    for i in range(vid_len):
        while vid_timestamps[i] > imu_timestamps[j] \
          and j < imu_len - 1:
            j += 1
        indices[i] = j
        
    return indices


def loadImuData(t_init, devices):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [int] t_init:
    [list(str)] devices:
    
    Returns:
    --------
    [list(np array)] imus:
    """
    
    # Load IMU data
    fn = os.path.join('data', 'imu', '{}.csv'.format(t_init))
    data = np.loadtxt(fn, delimiter=',')
    
    # Infer the length of a sample and make sure it evenly divides the number
    # of recorded variables
    num_samples, num_vars = data.shape
    sample_len = int(num_vars / len(devices))
    assert(num_vars % sample_len == 0)
    
    # Slice each IMU's data from the main array into a list
    imus = []
    for i, dev in enumerate(devices):
        i_start = i * sample_len
        i_end = (i + 1) * sample_len
        imus.append(data[:,i_start:i_end])
    
    # Convert IMU data to natural units
    # FIXME: Get conversion factors from device settings file
    for imu in imus:
        # Calculate time relative to TRIAL start (this is before each IMU
        # actually started streaming data)
        imu[:,0] = imu[:,0] - t_init
        
        # Accelerometer range setting +/- 8g --> divide by 4096 to get units of
        # g (ie acceleration due to earth's gravitational field)
        imu[:,2:5] = imu[:,2:5] / 4096.0
        
        # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get
        # units of degrees per second --> divide by 180 to get radians / sec
        imu[:,5:8] = imu[:,5:8] * 0.07 / 180
        
        # Multiply by 1e-4 to get units of mT, but this will be very
        # approximate (see WAX9 developer guide)
        imu[:,8:11] = imu[:,8:11] * 1e-4
    
    return imus


def loadLabels(t_init):
    """
    Try to load annotations from file for trial starting at time t_init. If
    there are no labels, warn the user and return an empty list.
    
    Args:
    -----
    [int] t_init:
    
    Returns:
    --------
    [np array] labels: Structured array with the following fields
      [int] start: Video frame index for start of event
      [int] end: Video frame index for end of event
      [int] action: Integer event ID (0-5). See labels file for description
      [int] object: Integer ID for object of event. Ex: 1 in 'place 1 on
        4'. Not all events have objects; when object N/A ID is 0
      [int] target: Integer ID for target of event. Ex: 4 in 'place 1 on
        4'. Not all events have targets; when target is N/A ID is 0
      [str] obj_studs: Adjacent studs are recorded for placement events. The
        i-th entry in this list is adjacent to the i-th entry in tgt_studs.
      [str] tgt_studs: Same as above. The i-th entry in this list is adjacent
        to the i-th entry in obj_studs.
    """
    
    fn = os.path.join('data', 'labels', '{}.csv'.format(t_init))
    
    if not os.path.exists(fn):
        print('Trial {}: no labels found'.format(t_init))
        return []
    
    # 'studs' are length-18 strings because 18 is an upper bound on the
    # description length for this attribute: max 8 studs + 8 spaces + 2 parens
    typestruct = [('start', 'i'), ('end', 'i'), ('action', 'i'),
                  ('object', 'i'), ('target', 'i'), ('obj_studs', 'S18'),
                  ('tgt_studs', 'S18')]
    labels = np.loadtxt(fn, delimiter=',', dtype=typestruct)
    return labels


def parseLabels(labels, num_frames):
    """
    Return boundaries and labels for annotated actions.
    
    Args:
    -----
    [np array] labels:
    [int] num_frames:
    
    Returns:
    --------
    [np array] bounds:
    [np array] actions:
    """
    
    bounds = []
    actions = []
    
    prev_start = 0
    prev_end = 0
    prev_action = 0
    bounds.append(prev_end)
    for i in range(len(labels)):
        
        # Let's define a (start index, end index, action) triple as an 'event'
        cur_start = labels['start'][i]
        cur_end = labels['end'][i]
        cur_action = labels['action'][i]
        
        # Sometimes a block is placed adjacent to multiple other blocks, so we
        # have multiple instances of the same event. Skip the duplicate
        # versions because we can't represent them in the visualization (yet)
        if cur_start == prev_start and cur_end == prev_end \
        and cur_action == prev_action:
            continue
        
        # Label indices should increase monotonically, assuming we skipped
        # duplicate events
        assert(cur_start >= prev_end)
        
        # If there's a gap between end of last action and the beginning of the
        # current action, fill it with inactivity
        if cur_start > prev_end:
            actions.append(0)   # 0 means inactive
            bounds.append(cur_start)
        
        # Append the current action and its end index
        actions.append(cur_action)
        bounds.append(cur_end)
        
        # Update previous event info (kind of unnecessary since we can just
        # index at i-1 if we want)
        prev_start = cur_start
        prev_end = cur_end
        prev_action = cur_action
        
    # Write inactivity until end if the last index isn't labeled
    assert(num_frames - 1 >= prev_end)
    if num_frames - 1 > prev_end:
        actions.append(0)
        bounds.append(num_frames - 1)
    
    return (np.array(bounds), np.array(actions))


def plotImuData(t_init, devices):
    """
    Save plots of IMU acceleration, angular velocity, magnetic field
    (With annotations in the background, if they exist)
    
    Args:
    -----
    [int] t_init:
    [list(str)] devices:
    """
    
    # Load IMU data
    imus = loadImuData(t_init, devices)
    
    # Load RGB frame timestamps
    fn = os.path.join('data', 'rgb', str(t_init), 'frame-timestamps.csv')
    video_timestamps = np.loadtxt(fn) - t_init
    num_vid_frames = video_timestamps.shape[0]
    
    # Load labels (empty list if no labels)
    label_names = ('inactive', 'remove', 'rotate', 'rotate all',
                   'place above', 'place below', 'place beside')
    num_labels = len(label_names)
    labels = loadLabels(t_init)
    
    # Define the output directory (for saving figures) and create it if it
    # doosn't already exist
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    titles = ['Acceleration', 'Angular velocity', 'Magnetic field']
    syms = ['a', '\omega', 'B']
    units = ['g', '$\pi$ rad/s', 'mT']
    abbrevs = ['accel', 'ang-vel', 'mag']
    coords = ['x', 'y', 'z']
    assert(len(coords) == 3)
    
    # Set colormaps and normalization for plotting labels
    cmap = mpl.cm.Pastel2
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('custom', cmap_list, cmap.N)
    bounds = list(range(num_labels))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Convert readings to natural units and plot, for each IMU selected
    selected_devices = range(len(imus))
    for i in selected_devices:
        
        imu = imus[i]
        name = devices[i]
        
        # Filter out bad data for now b/c it makes the plots impossible to read
        # (bad data are marked with a timestamp of 0)
        bad_data = np.less(imu[:,0], 1.0)
        imu = imu[np.logical_not(bad_data),:]
        
        # Get action boundaries (in video frames) from annotations and convert
        # to imu frames
        vid_bounds, actions = parseLabels(labels, num_vid_frames)
        vid2imu = vidFrame2imuFrame(video_timestamps, imu[:,0])
        imu_bounds = vid2imu[vid_bounds]
        
        # Plot accelerometer, gyroscope, magnetometer readings
        for j, title in enumerate(titles):
            
            start = 2 + j * len(coords)
                                    
            # X, Y, Z readings go in separate subplots on the same figure
            # Make an extra subplot for label colors
            f, axes = plt.subplots(len(coords)+1, 1, figsize=(6, 6.5))
            for k, ax in enumerate(axes[:-1]):
                
                # Plot IMU reading
                ax.plot(imu[:,0], imu[:,start+k], color='black', label=name,
                        zorder=3)
                ax.scatter(imu[:,0], imu[:,start+k], color='red', marker='.',
                           label=name, zorder=2)
                
                # Plot labels as colorbar in background, but only if they exist
                if imu_bounds.shape[0] > 0:
                    max_val = np.max(imu[:,start+k])
                    min_val = np.min(imu[:,start+k])
                    ax.pcolor(imu[imu_bounds,0], np.array([min_val, max_val]),
                              np.tile(actions, (2,1)), cmap=cmap, norm=norm,
                              zorder=1)
                
                # Label X and Y axes
                ax.set_xlabel('t (s)')
                fmt_str = r'${}_{}$ ({})'
                ax.set_ylabel(fmt_str.format(syms[j], coords[k], units[j]))
            
            # Plot colormap used for labels with tics at label indices
            mpl.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm,
                                      orientation='horizontal', ticks=bounds,
                                      boundaries=bounds)
            axes[0].set_title('{} in IMU frame, {}'.format(title, name))
            f.tight_layout()
            
            # Save plot and don't show
            fname = '{}_{}.pdf'.format(abbrevs[j], name)
            f.savefig(os.path.join(imu_fig_path, fname))
            plt.close()


def trackIMU(imu_data):
    """
    Determine a block's orientation and position from the IMU accelerometer,
    gyroscope, and magnetometer measurements
    
    Args:
    -----
    [np array] imu_data: n-by-11 array (where n is the number of samples in the
      trial) containing IMU sensor readings from one device. Columns are
      [0] ---- Universal relative time, measured from t_init (seconds)
      [1] ---- Sensor time (no units)
      [2:4] -- Acceleration [xyz] (g * meters / second^2)
      [5:7] -- Angular velocity [xyz] (radians / second)
      [8:10] - Magnetic field [xyz] (milliTeslas)
    
    Returns:
    --------
    [np array] R: Block orientation (flattened rotation matrix) at each
      sample. n-by-9 array. Matrix is row-major flattened.
    """
    
    R = np.zeros((imu_data.shape[0], 9))
    V = np.zeros((imu_data.shape[0], 3))
    X = np.zeros((imu_data.shape[0], 3))
    
    t = imu_data[:,0]
    a = imu_data[:,2:5]
    w = imu_data[:,5:8]
    
    # Strapdown inertial navigation (see "An introduction to inertial
    # navigation", Oliver J. Woodman, section 6)
    
    # Assume block is initially at (1, 1, 1), not moving, rotated 0 degrees
    # w/r/t the global reference frame
    Rt = np.identity(3)
    Xt = np.array([1, 1, 1])
    Vt = np.array([0, 0, 0])
    R[0,:] = Rt.ravel()
    V[0,:] = Vt
    X[0,:] = Xt
    
    # Acceleration due to gravity (in global frame), units are ~9.81 * m/s^2
    # This is correct as long as the assumption about R(t0) above is correct
    g = np.array([0, 0, -1])
    
    # FIXME: This can be calculated more efficiently
    for i in range(t.shape[0] - 1):
        dt = t[i+1] - t[i]
        #---[ORIENTATION]------------------------------
        # Integral of system matrix
        B = np.array([[   0   , -w[i,2],  w[i,1]],
                      [ w[i,2],    0   , -w[i,0]],
                      [-w[i,1],  w[i,0],    0   ]]) \
          * dt
        
        # 2-norm of B
        sigma = np.sqrt(w[i,:].dot(w[i,:])) * np.abs(dt)
        
        # Estimated transition matrix
        A = np.identity(3)
        if sigma > 0:
            A += np.sin(sigma) / sigma * B \
               + (1 - np.cos(sigma)) / (sigma ** 2) * B.dot(B)
        
        # New orientation
        Rt = Rt.dot(A)
        R[i+1,:] = Rt.ravel()
        
        #---[POSITION]------------------------------
        # Project acceleration onto global reference frame and convert to
        # meters / second^2
        a_global = Rt.dot(a[i,:]) * 9.81
        
        # Double-integrate to get position
        V[i+1,:] = V[i,:] + (a_global + g) * dt
        X[i+1,:] = X[i,:] + V[i+1,:] * dt
        
    return (R, V, X)


def plotPosition(X):
    """
    3D plot of IMU location
    
    Args:
    -----
    [np array] X: IMU position (n-by-3, where n is the number of
      samples in the trial)
    
    Returns:
    --------
    [int] fig: Figure handle for plot
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    max_dist = X.ravel().max()
    min_dist = X.ravel().min()
    ax.scatter(X[:,0], X[:,1], X[:,2], c=np.arange(X.shape[0]),
               cmap=mpl.cm.Spectral)
    ax.set_xlim(min_dist, max_dist)
    ax.set_ylim(min_dist, max_dist)
    ax.set_zlim(min_dist, max_dist)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Sensor Position')
    ax.legend()
    
    return fig

if __name__ == '__main__':
    t_init = 1458670623
    devices = ('08F1', '095D', '090F', '0949')
    #plotImuData(t_init, devices)
    
    imus = loadImuData(t_init, devices)
    R, V, X = trackIMU(imus[0])
    fig = plotPosition(X)
    
    