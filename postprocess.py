"""
postprocess.py
  Plot recorded IMU data (for now)

AUTHOR
  Jonathan D. Jones
"""


import os
import numpy as np
from matplotlib import pyplot as plt


def vidFrame2imuFrame(video_timestamps, kinem_timestamps):
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

def loadLabels(t_init):
    """
    Try to load annotations from file for trial starting at time t_init. If
    there are no labels, warn the user and return None.
    
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
        return None
    
    # 'studs' are length-18 strings because 18 is an upper bound on the
    # description length for this attribute: max 8 studs + 8 spaces + 2 parens
    typestruct = [('start', 'i'), ('end', 'i'), ('action', 'i'),
                  ('object', 'i'), ('target', 'i'), ('obj_studs', 'S18'),
                  ('tgt_studs', 'S18')]
    labels = np.loadtxt(fn, delimiter=',', dtype=typestruct, skiprows=2)
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
        assert(cur_start >= prev_end)
        if cur_start > prev_end:
            actions.append(0)   # 0 means inactive
            bounds.append(cur_start)
        actions.append(cur_action)
        bounds.append(cur_end)
        prev_end = cur_end
    """
    # Write inactivity until end if prev_end isn't the last index
    assert(num_frames - 1 >= prev_end)
    if num_frames - 1 > prev_end:
        actions.append(0)
        bounds.append(num_frames - 1)
    """
    
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
    labels = loadLabels(t_init)
    if labels is None:
        return
    
    imu_fig_path = os.path.join('output', 'figures', 'imu', str(t_init))
    if not os.path.exists(imu_fig_path):
        os.makedirs(imu_fig_path)
    
    # Plot accelerometer, gyroscope, magnetometer readings
    selected_devices = range(len(imus))
    for i in selected_devices:
        
        imu = imus[i]
        name = devices[i]
        
        # FIXME: searching only the object and target fields for matching names
        #   ignores the rotate_all label
        # relevant_labels = np.logical_or(labels['object'] == i + 1,
        #                                 labels['target'] == i + 1)
        
        vid_bounds, actions = parseLabels(labels, num_vid_frames)
        # Convert video frame labels to frame indices for this IMU
        vid2imu = vidFrame2imuFrame(video_timestamps, imu[:,0])
        imu_bounds = vid2imu[vid_bounds] #[relevant_labels]]
    
        # Filter out bad data for now b/c it makes the plots impossible to read
        bad_data = np.less(imu[:,0], 1.0)
        imu = imu[np.logical_not(bad_data),:]
    
        # Calculate time relative to trial start
        imu[:,0] = imu[:,0] - t_init
    
        # Accelerometer range setting +/- 8g --> divide by 4096 to get units of g
        # Gyroscope range setting +/- 2000 dps --> multiply by 0.07 to get units
        # of degrees per second
        # Multiply by 1e-4 to get units of mT, but this will be very approximate
        # (see WAX9 developer guide)
        coeffs = [1 / 4096.0, 0.07 / 180, 1e-4]
        titles = ['Acceleration', 'Angular velocity', 'Magnetic field']
        syms = ['a', '\omega', 'B']
        units = ['g', '$\pi$ rad/s', 'mT']
        abbrevs = ['accel', 'ang-vel', 'mag']
        coords = ['x', 'y', 'z']
        assert(len(coords) == 3)
        
        for i, title in enumerate(titles):
            start = 2 + i * len(coords)
            end = 2 + (i + 1) * len(coords)
            imu[:,start:end] = imu[:,start:end] * coeffs[i]
            f, axes = plt.subplots(len(coords), 1)
            for j, ax in enumerate(axes):
                # Plot IMU reading
                ax.plot(imu[:,0], imu[:,start+j], c='black', label=name)
                # Plot labels as colorbar in background
                max_val = np.max(imu[:,start+j])
                min_val = np.min(imu[:,start+j])
                ax.pcolor(imu[imu_bounds,0], np.array([min_val, max_val]),
                          np.tile(actions, (2,1)), cmap='Pastel2')
                # Label X and Y axes
                ax.set_xlabel('t (s)')
                ax.set_ylabel(r'${}_{}$ ({})'.format(syms[i], coords[j], units[i]))
            axes[0].set_title('{} in IMU frame'.format(title))
            f.tight_layout()
            
            fname = '{}_{}.pdf'.format(abbrevs[i], name)
            f.savefig(os.path.join(imu_fig_path, fname))
            plt.close()
        
        """
        # Shrink plot width by 20% to make room for the legend
        for axes in (ax_a, ax_g, ax_m):
            for ax in axes:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        """


if __name__ == '__main__':
    t_init = 1457987953
    #labels = loadLabels(t_init)
    devices = ('08F1', '095D', '090F', '0949')
    plotImuData(t_init, devices, 11)