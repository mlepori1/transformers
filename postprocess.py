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


def rgbFrame2imuFrame(rgb_timestamps, imu_timestamps):
    """
    Find the indices of the IMU timestamps that most closely
    match the video timestamps.
    
    Args:
    -----
    [np vector] rgb_timestamps: The i-th entry contains the universal time
      at which the i-th VIDEO FRAME occurred
    [np vector] imu_timestamps: The i-th entry contains the universal time
      at which the i-th IMU SAMPLE occurred
      
    Returns:
    --------
    [np vector] indices: Same length as vid_timestamps.
      The i-th entry holds the index of imu_timestamps that most
      closely matches the time of the i-th index in video_timestamps.
    """
    
    rgb_len = rgb_timestamps.shape[0]
    imu_len = imu_timestamps.shape[0]
            
    # For each video timestamp, find the index of the nearest
    # IMU timestamp (+/- one frame)
    # Video and IMU timestamps are in increasing order.
    indices = np.zeros((rgb_len,), dtype=int)
    j = 0
    for i in range(rgb_len):
        while rgb_timestamps[i] > imu_timestamps[j] \
          and j < imu_len - 1:
            j += 1
        indices[i] = j
        
    return indices


def loadImuData(t_init, devices):
    """
    Load IMU samples from the file specified by t_init
    
    Args:
    -----
    [int] t_init: Trial identifier. This is the (truncated) Unix time when the
      trial's data directory was created.
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
    for i in range(len(imus)):
        
        imu_data = imus[i]
        
        # Filter out all-zero rows representing missing samples
        # Missing samples can be identified because their timestamps are zero
        # instead of UNIX time
        # FIXME: Mark the error column when these rows are written in the
        #   first place
        bad_sample = imu_data[:,0] < 1.0
        imu_data = imu_data[np.logical_not(bad_sample),:]
        
        # Accelerometer range setting +/- 8g --> divide by 4096 to get units of
        # g (ie acceleration due to earth's gravitational field)
        imu_data[:,4:7] = imu_data[:,4:7] / 4096.0
        
        # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get
        # units of degrees per second --> divide by 180 to get radians / sec
        imu_data[:,7:10] = imu_data[:,7:10] * 0.07 / 180
        
        # Multiply by 1e-4 to get units of mT, but this will be very
        # approximate (see WAX9 developer guide)
        imu_data[:,10:13] = imu_data[:,10:13] * 1e-4
        
        imus[i] = imu_data
    
    return imus

def loadRgbFrameTimestamps(t_init):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [int] t_init: Trial identifier. This is the (truncated) Unix time when the
      trial's data directory was created.
    
    Returns:
    --------
    [np vector] rgb_timestamps:
    """
    
    fn = os.path.join('data', 'rgb', str(t_init), 'frame-timestamps.csv')
    rgb_timestamps = np.loadtxt(fn)
    
    return rgb_timestamps


def loadLabels(t_init):
    """
    Try to load annotations from file for trial starting at time t_init. If
    there are no labels, warn the user and return an empty list.
    
    Args:
    -----
    [int] t_init: Trial identifier. This is the (truncated) Unix time when the
      trial's data directory was created.
    
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
    Return event labels of annotated actions, along with the RGB frame indices
    of action boundaries
    
    Args:
    -----
    [np array] labels: Structured array of event annotations (see loadLabels)
    [int] num_frames: Number of RGB frames recorded
    
    Returns:
    --------
    [np vector] actions: Sequence of integer event labels for this trial
    [np vector] action_bounds: boundary indices for the actions specified
      above. The length of this vector is len(labels) + 1
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
    
    return (np.array(actions), np.array(bounds))


def imuActionBounds(labels, t_rgb, t_imu):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [np array] labels: Structured array of event annotations (see loadLabels)
    [np vector] t_rgb:
    [np vector] t_imu:
    
    Returns:
    --------
    [np vector] actions: Sequence of integer event labels for this trial
    [np vector] action_bounds: boundary indices for the actions specified
      above. The length of this vector is len(labels) + 1
    """
    
    num_rgb_frames = t_rgb.shape[0]
    
    # Get action boundaries (in video frames) from annotations and convert
    # to imu frames
    actions, rgb_bounds = parseLabels(labels, num_rgb_frames)
    rgb2imu = rgbFrame2imuFrame(t_rgb, t_imu)
    action_bounds = rgb2imu[rgb_bounds]
    
    return actions, action_bounds


def plot3dof(data, actions, action_bounds, fig_text):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [np array] data: n-by-4 array (n is the number of samples) with the
      following columns
      [0] - time
      [1] - x component
      [2] - y component
      [3] - z component
    [np vector] actions: Sequence of integer event labels for this trial
    [np vector] action_bounds: boundary indices for the actions specified
      above. The length of this vector is len(labels) + 1
    [list(str)] fig_text: List of text strings to be used in labeling the
      figure. Elements are the following
      [0] - title
      [1] - symbol
      [2] - unit
    
    Returns:
    --------
    [int] f: matplotlib handle for the figure that was created
    """
    
    # FIXME: don't use a magic number like this
    NUM_LABELS = 7
    
    assert(data.shape[1] == 4)
    assert(len(action_bounds) == len(actions) + 1)
    
    # Unpack figure text
    assert(len(fig_text) == 3)
    title = fig_text[0]
    sym = fig_text[1]
    unit = fig_text[2]
    
    # Set colormaps and normalization for plotting labels
    cmap = mpl.cm.Pastel2
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('custom', cmap_list, cmap.N)
    bounds = list(range(NUM_LABELS))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    coords = ['x', 'y', 'z']
                                    
    # X, Y, Z readings go in separate subplots on the same figure
    # Make an extra subplot for label colors
    f, axes = plt.subplots(len(coords)+1, 1, figsize=(6, 6.5))
    for k, ax in enumerate(axes[:-1]):
        
        # Plot IMU reading
        ax.plot(data[:,0], data[:,k+1], color='black', zorder=3)
        ax.scatter(data[:,0], data[:,k+1], color='red', marker='.', zorder=2)
        
        # Plot labels as colorbar in background, but only if they exist
        if action_bounds.shape[0] > 0:
            max_val = np.max(data[:,k+1])
            min_val = np.min(data[:,k+1])
            ax.pcolor(data[action_bounds,0], np.array([min_val, max_val]),
                      np.tile(actions, (2,1)), cmap=cmap, norm=norm,
                      zorder=1)
        
        # Label X and Y axes
        ax.set_xlabel(r'$t \/ (\mathrm{s})$')
        fmt_str = r'${}_{} \/ ({})$'
        ax.set_ylabel(fmt_str.format(sym, coords[k], unit))
    
    # Plot colormap used for labels with tics at label indices
    mpl.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm,
                              orientation='horizontal', ticks=bounds,
                              boundaries=bounds)
    axes[0].set_title('{} in IMU frame'.format(title))
    f.tight_layout()
    
    return f


def plotImuData(t_init, devices):
    """
    Save plots of IMU acceleration, angular velocity, magnetic field
    (With annotations in the background, if they exist)
    
    Args:
    -----
    [int] t_init:
    [list(str)] devices:
    
    Returns:
    --------
      (Nothing)
    """
    
    # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
    imus = loadImuData(t_init, devices)
    rgb_timestamps = loadRgbFrameTimestamps(t_init)
    labels = loadLabels(t_init)
    
    # Define the output directory (for saving figures) and create it if it
    # doesn't already exist
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    fig_txt = [('Acceleration', 'a', 'g \/ \mathrm{m/s}^2'),
               ('Angular velocity', '\omega', '\pi \/ \mathrm{rad/s}'),
               ('Magnetic field', 'B', '\mathrm{mT}')]
    fn_abbrevs = ['accel', 'ang-vel', 'mag']
    
    norm_text = ('Square L2 norm', '\| \cdot \|', ' ')
    
    # Convert readings to natural units and plot, for each IMU selected
    selected_devices = range(len(imus))
    for i in selected_devices:
        
        imu = imus[i]
        name = devices[i]
        
        actions, imu_bounds = imuActionBounds(labels, rgb_timestamps, imu[:,0])
        
        # Plot accelerometer, gyroscope, magnetometer readings
        norm_data = imu[:,0:1]
        for j, title in enumerate(fig_txt):
            
            # Make plot of current 3-DOF data
            start = 4 + j * 3
            end = start + 3
            data = np.hstack((imu[:,0:1], imu[:,start:end]))
            f = plot3dof(data, actions, imu_bounds, fig_txt[j])
            
            # Save plot and don't show
            fname = '{}_{}.pdf'.format(fn_abbrevs[j], name)
            f.savefig(os.path.join(imu_fig_path, fname))
            plt.close()
            
            # Append magnitude
            sq2norm = np.sum(imu[:,start:end] * imu[:,start:end], 1)
            norm_data = np.vstack((norm_data.T, sq2norm)).T
        
        # Plot, save, and close magnitude figures
        f = plot3dof(norm_data, actions, imu_bounds, norm_text)
        fname = 'norms_{}.pdf'.format(name)
        f.savefig(os.path.join(imu_fig_path, fname))
        plt.close()
    
    return


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
    [np array] A_g: Block acceleration (in global frame) at each sample. n-by-3
      array.
    [np array] V: Block velocity at each sample. n-by-3 array.
    [np array] S: Block position at each sample. n-by-3 array.
    """
    
    R = np.zeros((imu_data.shape[0], 9))
    A_g = np.zeros((imu_data.shape[0], 3))
    V = np.zeros((imu_data.shape[0], 3))
    S = np.zeros((imu_data.shape[0], 3))
    
    t = imu_data[:,0]
    a = imu_data[:,4:7]
    w = imu_data[:,7:10]
    
    # Strapdown inertial navigation (see "An introduction to inertial
    # navigation", Oliver J. Woodman, section 6)
    
    # Assume block is initially at (1, 1, 1), not moving, rotated 0 degrees
    # w/r/t the global reference frame
    Rt = np.identity(3)
    St = np.array([1, 1, 1])
    Vt = np.array([0, 0, 0])
    R[0,:] = Rt.ravel()
    V[0,:] = Vt
    S[0,:] = St
    
    # Acceleration due to gravity (in global frame), units are ~9.81 * m/s^2
    # This is correct as long as the assumption about R(t0) above is correct
    # Otherwise this should be projected onto the global axes (?)
    g = np.array([0, 0, -1])
    
    # FIXME: This can be calculated more efficiently
    a_global_prev = a[0,:]
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
        # FIXME: This needs a name that isn't A
        A = np.identity(3)
        if sigma > 0:
            A += np.sin(sigma) / sigma * B \
               + (1 - np.cos(sigma)) / (sigma ** 2) * B.dot(B)
        
        # New orientation
        Rt = Rt.dot(A)
        R[i+1,:] = Rt.ravel()
        
        #---[POSITION]------------------------------
        # Project acceleration onto global reference frame and account for the
        # effect of gravity (we're still in units of g m/s^2)
        a_global = (Rt.dot(a[i,:]) + g)
        A_g[i] = a_global
                
        # Double-integrate to get position
        # Assume velocity is zero if change in acceleration is near zero
        a_diff = a_global_prev - a_global
        A_THRESH = 0.01
        if np.sqrt(a_diff.dot(a_diff)) < A_THRESH:
            V[i+1,:] = np.zeros(3)
        else:
            V[i+1,:] = V[i,:] + a_global * 9.81 * dt
        
        S[i+1,:] = S[i,:] + V[i+1,:] * dt
        
        a_global_prev = a_global
        
    return (R, A_g, V, S)


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
               cmap=mpl.cm.Spectral, edgecolors='face')
    ax.set_xlim(min_dist, max_dist)
    ax.set_ylim(min_dist, max_dist)
    ax.set_zlim(min_dist, max_dist)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Sensor Position')
    
    return fig


def plotKinematics(t_init, devices):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [int] t_init:
    [list(str)] devices:
    
    Returns:
    --------
      (Nothing)
    """    
    
    # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
    imus = loadImuData(t_init, devices)
    rgb_timestamps = loadRgbFrameTimestamps(t_init)    
    labels = loadLabels(t_init)
    
    # Define the output directory (for saving figures) and create it if it
    # doesn't already exist
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    fn_abbrevs = ('global-accel', 'vel', 'pos')
    text = [('Global acceleration', 'a', 'g \/ \mathrm{m/s}^2'),
            ('Velocity', 'v', '\mathrm{m/s}'),
            ('Position', 's', '\mathrm{m}')]
    
    # Convert readings to natural units and plot, for each IMU selected
    selected_devices = range(len(imus))
    for i in selected_devices:
        
        imu = imus[i]
        name = devices[i]

        actions, imu_bounds = imuActionBounds(labels, rgb_timestamps, imu[:,0])
        
        R, A_g, V, S = trackIMU(imu)

        for j, data in enumerate((A_g, V, S)):
            augmented_data = np.hstack((imu[:,0:1], data))        
            f = plot3dof(augmented_data, actions, imu_bounds, text[j])
            
            # Save plot and don't show
            fname = '{}_{}.pdf'.format(fn_abbrevs[j], name)
            f.savefig(os.path.join(imu_fig_path, fname))
            plt.close()
            
        f = plotPosition(S)
    
    return


def pointwiseCorrelation(t_init, devices):
    """
    Calculate pointwise cosine similarity for all pairs of devices in imus.
    
    Args:
    -----
    [int] t_init:
    [list(str)] devices:
    
    Returns:
    --------
      (Nothing)
    """
    
    NUM_LABELS = 7
    
    # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
    imus = loadImuData(t_init, devices)
    rgb_timestamps = loadRgbFrameTimestamps(t_init)    
    labels = loadLabels(t_init)
    
    # Define the output directory (for saving figures) and create it if it
    # doesn't already exist
    corr_fig_path = os.path.join('output', 'figures', 'imu-corrs', str(t_init))
    if not os.path.exists(corr_fig_path):
        os.makedirs(corr_fig_path)
    
    # Set colormaps and normalization for plotting labels
    cmap = mpl.cm.Pastel2
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('custom', cmap_list, cmap.N)
    bounds = list(range(NUM_LABELS))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    for i, imu1 in enumerate(imus):
        
        name1 = devices[i]
        
        # Correlation is symmetric, so only calculate for the n(n-1)/2 unique
        # cases
        for j in range(i+1, len(imus)):
            
            imu2 = imus[j]
            name2 = devices[j]
            
            syms = ('a', '\omega', 'B')
            f, axes = plt.subplots(len(syms) + 1, 1, figsize=(6, 6))
            
            # Calculate cosine distance = (x^T y) / (|x| |y|) for acceleration,
            # angular velocity, and magnitude readings
            for k, ax in enumerate(axes[:-1]):
                
                start = 4 + k * 3
                end = start + 3
                
                data1 = imu1[:, start:end]
                data2 = imu2[:, start:end]
                
                norm1 = np.sqrt(np.sum(data1 * data1, 1))
                norm2 = np.sqrt(np.sum(data2 * data2, 1))
                
                ptwise_corr = np.sum(data1 * data2, 1) / (norm1 * norm2)
                
                # Filter out bad data for now b/c it makes the plots impossible to read
                # (bad data are marked with a timestamp of 0)
                bad_data1 = np.less(imu1[:,0], 1.0)
                bad_data2 = np.less(imu2[:,0], 1.0)
                bad_data = np.logical_or(bad_data1, bad_data2)
                ptwise_corr = ptwise_corr[np.logical_not(bad_data)]
                
                # Calculate time relative to TRIAL start (this is before each IMU
                # actually started streaming data)
                t_imu1 = imu1[np.logical_not(bad_data),0] - t_init
                
                actions, imu_bounds = imuActionBounds(labels, rgb_timestamps, t_imu1)
                
                ax.plot(t_imu1, ptwise_corr, color='k') #, zorder=2)
                
                # Plot labels as colorbar in background, but only if they exist
                if imu_bounds.shape[0] > 0:
                    max_val = ptwise_corr.max()
                    min_val = ptwise_corr.min()
                    ax.pcolor(t_imu1, np.array([min_val, max_val]),
                              np.tile(labels, (2,1)), cmap=cmap, norm=norm) #,
                              #zorder=1)
                
                ax.set_ylabel(r'$\cos \theta_{}$'.format(syms[k]))
                ax.set_xlabel(r'$t \/ (\mathrm{s})$')
            
            # Plot colormap used for labels with tics at label indices
            mpl.colorbar.ColorbarBase(axes[-1], cmap=cmap, norm=norm,
                                      orientation='horizontal', ticks=bounds,
                                      boundaries=bounds)
            axes[0].set_title('Pointwise cosine similarity for IMU readings')
            f.tight_layout()
            
            # Save plot and don't show
            fname = 'cos-sim_{}-{}.pdf'.format(name1, name2)
            f.savefig(os.path.join(corr_fig_path, fname))
            plt.close()
    
    return


if __name__ == '__main__':
    
    import glob
    
    # Batch process all imu data files
    devices = ('08F1', '095D', '090F', '0949')
    pattern = '/Users/jonathan/repo/blocks/data/imu/*.csv'
    for match in glob.glob(pattern):
        
        # match is a complete path. Split off the base path and file extension
        # to get t_init
        _, fn = os.path.split(match)
        t, _ = os.path.splitext(fn)
        t = int(t)
        
        plotImuData(t, devices)
        plotKinematics(t, devices)
        pointwiseCorrelation(t, devices)

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