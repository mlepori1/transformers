# -*- coding: utf-8 -*-
"""
libduplo.py
  Library of helper functions for the duplo corpus

AUTHOR
  Jonathan D. Jones
"""

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


def parseActions(labels):
    """
    Re-construct the progression of block states from action label sequence.
    
    Args:
    -----
    [np array] labels: (See loadlabels in duplocorpus.py)
    
    Returns:
    --------
    [list(gv Digraph)] states: List of graphviz graphs representing the
      progression of block states
    """
    
    import graphviz as gv
    
    # Return the state sequence in this list
    states = []
    
    # Sort actions in increasing order by their end index
    labels = np.sort(labels, order='end')
    
    # Create a directed graph representing the block construction and add all
    # 8 blocks as nodes
    state = gv.Digraph(name=str(len(states)), format='png')
    colors = ('red', 'yellow', 'green', 'blue')
    shapes = ('Msquare', 'box')
    widths = ('0.5', '1')
    block_id = 0
    for cur_width, cur_shape in zip(widths, shapes):
        for cur_color in colors:
            block_id += 1
            state.node(str(block_id), height='0.5', width=cur_width,
                       color=cur_color, shape=cur_shape, style='filled')
    
    states.append(state)
    
    # LABELS
    #   0 -- (inactive)
    #   1 -- place [object] above [target]
    #   2 -- place [object] adjacent to [target]
    #   3 -- rotate [object]
    #   4 -- translate [object]
    #   5 -- remove [object]
    #   6 -- pick up [object]
    relevant_actions = (1, 2, 5)    # These actions change the state
    prev_state = states[-1]
    prev_end = -1
    for label in labels:
        
        action = label['action']        
        if not action in relevant_actions:
            continue
        
        end = label['end']; assert(end >= prev_end)        
        object_block = str(label['object'])
        target_block = str(label['target'])
        
        # Initialize a new state graph with the contents of the previous state.
        # I do it this way because the Digraph class doesn't have a copy
        # constructor.
        cur_state = gv.Digraph(name=str(len(states)), format='png')
        cur_state.body = list(prev_state.body)
        
        if action == 1:     # Add a directed edge
            cur_state.edge(object_block, target_block)
        elif action == 2:   # Add an undirected edge
            cur_state.edge(object_block, target_block, dir='none')
        elif action == 5:
            new_body = []
            for entry in cur_state.body:
                # Matches all edges originating from the removed block
                start_pattern = '\t\t{}'.format(object_block)
                # Matches all undirected edges leading to the removed block
                end_pattern = '-> {} [dir=none]'.format(object_block)
                if not (entry.startswith(start_pattern) or
                   entry.endswith(end_pattern)):
                    new_body.append(entry)
            cur_state.body = new_body
        
        # By only adding states with strictly increasing end indices, we ensure
        # that simultaneous actions are displayed simultaneously.
        if end > prev_end:
            states.append(cur_state)
        
        prev_state = cur_state
        prev_end = end
    
    return states

    