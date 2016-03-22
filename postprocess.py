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
      labels:
      num_frames:
    
    Returns:
    --------
      bounds:
      actions:
    """
    
    bounds = []
    actions = []
    
    prev_end = 0
    bounds.append(prev_end)
    for i in range(len(labels)):
        cur_start = labels['start'][i]
        cur_end = labels['end'][i]
        cur_action = labels['action'][i]
        # Label indices should increase monotonically
        assert(cur_start >= prev_end)
        # If there's a gap between end of last action and the beginning of the
        # current action, fill it with inactivity
        if cur_start > prev_end:
            actions.append(0)   # 0 means inactive
            bounds.append(cur_start)
        # Append the current action and its end index
        actions.append(cur_action)
        bounds.append(cur_end)
        prev_end = cur_end
    # Write inactivity until end if prev_end isn't the last index
    assert(num_frames - 1 >= prev_end)
    if num_frames - 1 > prev_end:
        actions.append(0)
        bounds.append(num_frames - 1)
    
    return (np.array(bounds), np.array(actions))


def plotImuData(t_init, devices, sample_len):
    """
    Create plots of IMU acceleration, angular velocity, magnetic field
    
    Args:
    -----
      [int] t_init:
      [list(str)] devices:
      [int] sample_len:
    """
    
    # Map device names to integer IDs
    name2id = {devices[i]:i for i in range(len(devices))}

    # Load IMU data
    fn = os.path.join('data', 'imu', '{}.csv'.format(t_init))
    data = np.loadtxt(fn, delimiter=',')
    num_samples, num_vars = data.shape
    assert(num_vars % sample_len == 0)
    
    # Determine the number of IMUs present in this trial and slice their data
    # into a list
    # FIXME: we already know how many IMUs there are, it's len(devices)
    imus = []
    for i in range(int(num_vars / sample_len)):
        i_start = i * sample_len
        i_end = (i + 1) * sample_len
        imus.append(data[:,i_start:i_end])
    
    # Load RGB frame timestamps
    fn = os.path.join('data', 'rgb', str(t_init), 'frame-timestamps.csv')
    video_timestamps = np.loadtxt(fn)
    num_vid_frames = video_timestamps.shape[0]
    
    # Load labels if we have them. If not, no point in going any further
    label_names = ('inactive', 'remove', 'rotate', 'rotate all',
                   'place above', 'place below', 'place beside')
    num_labels = len(label_names)
    labels = loadLabels(t_init)
    if labels is None:
        return
    
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    # Accelerometer range setting +/- 8g --> divide by 4096 to get units of g
    # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get units
    # of degrees per second
    # Multiply by 1e-4 to get units of mT, but this will be very approximate
    # (see WAX9 developer guide)
    convert_factors = [1 / 4096.0, 0.07 / 180, 1e-4]
    titles = ['Acceleration', 'Angular velocity', 'Magnetic field']
    syms = ['a', '\omega', 'B']
    units = ['g', '$\pi$ rad/s', 'mT']
    abbrevs = ['accel', 'ang-vel', 'mag']
    coords = ['x', 'y', 'z']
    assert(len(coords) == 3)
    
    # Set colormaps and normalization for plotting labels (between 0 and 6)
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
    
        # Calculate time relative to trial start
        imu[:,0] = imu[:,0] - t_init
    
        # Plot accelerometer, gyroscope, magnetometer readings
        for j, title in enumerate(titles):
            
            # Slice accel/gyro/mag data and convert to natural units
            start = 2 + j * len(coords)
            end = 2 + (j + 1) * len(coords)
            imu[:,start:end] = imu[:,start:end] * convert_factors[j]
            
            # X, Y, Z readings go in separate subplots on the same figure
            f, axes = plt.subplots(len(coords)+1, 1, figsize=(6, 6.5))
            for k, ax in enumerate(axes[:-1]):
                
                # Plot IMU reading
                ax.plot(imu[:,0], imu[:,start+k], color='black', label=name)
                
                # Plot labels as colorbar in background, but only if they exist
                if imu_bounds.shape[0] > 0:
                    max_val = np.max(imu[:,start+k])
                    min_val = np.min(imu[:,start+k])
                    ax.pcolor(imu[imu_bounds,0], np.array([min_val, max_val]),
                              np.tile(actions, (2,1)), cmap=cmap, norm=norm)
                
                # Label X and Y axes
                ax.set_xlabel('t (s)')
                ax.set_ylabel(r'${}_{}$ ({})'.format(syms[j], coords[k], units[j]))
                
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


if __name__ == '__main__':
    t_init = 1457987953
    #labels = loadLabels(t_init)
    devices = ('08F1', '095D', '090F', '0949')
    plotImuData(t_init, devices, 11)