# -*- coding: utf-8 -*-
"""
duplocorpus.py
  Wrapper class for block study data (IMU, video, labels, meta, etc)

AUTHOR
  Jonathan D. Jones
"""

import sys
import os
import logging
import csv
import glob
import subprocess

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from libduplo import *


class DuploCorpus:
    
    
    def __init__(self):
        self.initPaths()
        self.initLogger()
        
        # ID strings for the recording devices
        self.imu_ids = ('08F1', '095D', '090F', '0949')
        self.camera_ids = ('rgb',) # 'depth')
        
        self.initTypestructs()
        
        # Load metadata array
        self.meta_data = self.readMetaData()
    
    
    def initTypestructs(self):
        """
        Set up typestruct variables for saving and loading data
        """
        
        self.metadata_types = [('trial id', 'i4'), ('participant id', 'U10'), 
                               ('birth month', 'U3'), ('birth year', 'U4'),
                               ('gender', 'U15'), ('has labels', 'i4')]       \
                            + [(name, 'U10') for name in self.imu_ids]
        
        self.label_types = [('start', 'i'), ('end', 'i'), ('action', 'i'),
                            ('object', 'i'), ('target', 'i'),
                            ('obj_studs', 'S18'), ('tgt_studs', 'S18')]
    
    
    def initLogger(self):
        """
        Set up logger for warnings, etc
        """
        
        self.logger = logging.getLogger('blocks')
        # reset logger handlers, otherwise extra handlers get added every time
        # the script is run in an ipython console (I don't know what I'm doing)
        self.logger.handlers = []
        fn = os.path.join(self.paths['root'], 'blocks.log')
        fh = logging.FileHandler(fn)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        fmtstr = '%(asctime)s | %(levelname)s | %(message)s'
        formatter = logging.Formatter(fmtstr)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    
    def initPaths(self):
        """
        Build directory tree for the block study corpus
        """
        
        self.paths = {}
        
        # Root directory is one level above the directory that contains this
        # file
        self.paths['src'] = os.path.dirname(os.path.abspath(__file__))
        self.paths['root'] = os.path.dirname(self.paths['src'])
        self.paths['output'] = os.path.join(self.paths['root'], 'output')   
        self.paths['figures'] = os.path.join(self.paths['root'], 'figures')
        self.paths['working'] = os.path.join(self.paths['root'], 'working')
        self.paths['data'] = os.path.join(self.paths['root'], 'data')
        self.paths['imu-samples'] = os.path.join(self.paths['data'], 'imu-samples')
        self.paths['imu-settings'] = os.path.join(self.paths['data'], 'imu-settings')
        self.paths['video-frames'] = os.path.join(self.paths['data'], 'video-frames')
        self.paths['frame-timestamps'] = os.path.join(self.paths['data'], 'frame-timestamps')
        self.paths['raw'] = os.path.join(self.paths['data'], 'raw')
        self.paths['labels'] = os.path.join(self.paths['data'], 'labels')
        
        # Make directories if they don't exist
        for key in self.paths.keys():
            if not os.path.exists(self.paths[key]):
                os.makedirs(self.paths[key])
    
    
    def readMetaData(self):
        """
        Read metadata from file and return as a numpy structured
        array.
        
        Returns:
        --------
        [np array] meta_data: Numpy structured array with the following fields
          [int] trial id:
          [str] participant id:
          [str] birth month:
          [str] birth year:
          [str] gender:
          [int] has_labels:
          [str] 08F1:
          [str] 095D:
          [str] 090F:
          [str] 0949:
        """

        meta_data = []
        
        # Start a new, empty metadata array if a file doesn't already exist
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        if not os.path.exists(fn):
            return np.array(meta_data, dtype=self.metadata_types)
        
        # Read contents of the metadata file line-by-line
        with open(fn, 'r') as metafile:
            metareader = csv.reader(metafile)
            for row in metareader:
                meta_data.append(tuple(row))
        
        return np.array(meta_data, dtype=self.metadata_types)
    
    
    def writeMetaData(self):
        """
        Write this object's metadata array to a file names meta-data.csv in the
        data directory
        """
        
        # Write contents of the metadata array to file line-by-line
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        with open(fn, 'w') as metafile:
            metawriter = csv.writer(metafile)
            for row in self.meta_data:
                metawriter.writerow(row)
    
    
    def writeImuSettings(self, trial_id, imu_settings):
        """        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [np array] dev_settings:
        """
        
        # Write settings array to CSV line-by-line
        fn = os.path.join(self.paths['imu-settings'], '{}.csv'.format(trial_id))
        with open(fn, 'w') as settings_file:
            settings_writer = csv.writer(settings_file)
            for row in imu_settings:
                settings_writer.writerow(row)
    
    
    def readImuData(self, trial_id):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        
        Returns:
        --------
        [dict(str->np array)] imus:
        """
        
        imus = {}
        for imu_id in self.imu_ids:
            
            if self.meta_data[imu_id][trial_id] == 'UNUSED':
                continue
            
            fn = '{}-{}.csv'.format(trial_id, imu_id)
            path = os.path.join(self.paths['imu-samples'], fn)
            imu_data = np.loadtxt(path, delimiter=',')
            
            # FIXME: convert intelligently
            imu_data[:,3] = imu_data[:,3] / 65536.0   # Convert sample timestamp to seconds
            imu_data[:,4:7] = imu_data[:,4:7] / 4096.0    # Convert acceleration samples to g
            imu_data[:,7:10] = imu_data[:,7:10] * 0.07  # Convert angular velocity samples to deg/s
            imu_data[:,10:13] = imu_data[:,10:13] * 1e-3  # Convert magnetic field to units of mGauss
            
            imus[imu_id] = imu_data
        
        return imus


    def writeImuData(self, trial_id, imu_data):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [dict(str->np array)] imu_data:
        """
        
        for name, data in imu_data.items():
            fn = '{}-{}.csv'.format(trial_id, name)
            path = os.path.join(self.paths['imu-samples'], fn)
            sample_len = data.shape[1]
            fmtstr = ['%15f'] + (sample_len - 1) * ['%i']
            np.savetxt(path, data, delimiter=',', fmt=fmtstr)
    
    
    def readFrameTimestamps(self, trial_id):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
                
        Returns:
        --------
        [dict(str->np array)] frame_timestamps:
        """
        
        for camera_id in self.camera_ids:
            fn = '{}-{}.csv'.format(trial_id, camera_id)
            path = os.path.join(self.paths['frame-timestamps'], fn)
            frame_timestamps = np.loadtxt(path, delimiter=',')
        
        return frame_timestamps
        
    
    def writeFrameTimestamps(self, trial_id, frame_timestamps):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [dict(np array)] img_timestamps:
        """
        
        for name, timestamps in frame_timestamps.items():
            fn = '{}-{}.csv'.format(trial_id, name)
            path = os.path.join(self.paths['frame-timestamps'], fn)
            sample_len = timestamps.shape[1]
            fmtstr = ['%15f'] + (sample_len - 1) * ['%i']
            np.savetxt(path, timestamps, delimiter=',', fmt=fmtstr)
    
    
    def readLabels(self, trial_id):
        """
        Try to load annotations from file for trial starting at time t_init. If
        there are no labels, warn the user and return an empty list.
        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        
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
        
        fn = os.path.join(self.paths['labels'], '{}.csv'.format(trial_id))
        
        if not os.path.exists(fn):
            self.logger.warn('Trial {}: no labels found'.format(trial_id))
            return []
        
        # 'studs' are length-18 strings because 18 is an upper bound on the
        # description length for this attribute: max 8 studs + 8 spaces + 2 parens
        labels = np.loadtxt(fn, delimiter=',', dtype=self.label_types)
        return labels
    
    
    def parseRawData(self, trial_id):
        """
        Read raw data from file and convert to numpy arrays
    
        Args:
        -----
        [str] path: Path (full or relative) to raw data file
        [list(str)] imu_dev_names: List of IMU ID strings; define p as the number
          of items in this list
        [list(str)] img_dev_names: List of image capture device ID strings; define
          q as the number of items in this list
    
        Returns:
        --------
        [dict(str->np array)] imu_data: Size n-by-p*d, where n is the number of IMU
          samples, p is the number of IMU devices, and d is the length of one IMU
          sample. Each holds p length-d IMU samples, concatenated in the order
          seen in imu_dev_names
        [np array] img_data: Size m-by-p. Contains the global timestamp for every video
          frame recorded
        [int] sample_len: Length of one IMU sample (d)
        """
        
        imu_data = {}
        path = os.path.join(self.paths['raw'], '{}-imu.csv'.format(trial_id))
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                
                timestamp = float(row[0])
                dev_name = row[-1]
                
                if not dev_name in imu_data:
                    imu_data[dev_name] = []
    
                # IMU samples are one-indexed, so subtract 1 to convert to
                # python's zero-indexing
                sample_idx = int(row[2]) - 1
                sample = [timestamp] + [int(x) for x in row[1:-1]]
                
                # Don't try to interpret rows that represent bad samples
                error = int(row[1])
                if error:
                    continue
                
                # Append rows of zeros to the data matrix until the last row
                # is the sample index. Then write the current sample to its
                # place in the sample index.
                for i in range(sample_idx - (len(imu_data[dev_name]) - 1)):
                    dummy = [0.0] * len(sample)
                    dummy[1] = 1
                    imu_data[dev_name].append(dummy)
                imu_data[dev_name][sample_idx] = sample
        
        for key, entry in imu_data.items():
            imu_data[key] = np.array(entry)
        
        path = os.path.join(self.paths['raw'], '{}-timestamps.csv'.format(trial_id))
        frame_timestamps = {}
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                timestamp = float(row[0])
                dev_name = row[-1]
                
                if not dev_name in frame_timestamps:
                    frame_timestamps[dev_name] = []
                
                # Video frames are zero-indexed
                sample_idx = int(row[2])
                sample = [timestamp] + [int(x) for x in row[1:-1]]
                
                # Append rows of zeros to the data matrix until the last row
                # is the sample index. Then write the current sample to its
                # place in the sample index.
                for i in range(sample_idx - (len(frame_timestamps[dev_name]) - 1)):
                    dummy = [0.0] * len(sample)
                    dummy[1] = 1
                    frame_timestamps[dev_name].append(dummy)
                frame_timestamps[dev_name][sample_idx] = sample
        
        for key, entry in frame_timestamps.items():
            frame_timestamps[key] = np.array(entry)
        
        return (imu_data, frame_timestamps)
    
    
    def makeVideo(self, trial_id):
        """
        Convert frames to avi video
        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        """
        
        # NOTE: This depends on the avconv utility for now
        print('')
        frame_fmt = os.path.join(self.paths['video-frames'], '{}-rgb'.format(trial_id), '%6d.png')
        video_path = os.path.join(self.paths['video-frames'], '{}-rgb.avi'.format(trial_id))
        make_video = ['avconv', '-y', '-f', 'image2', '-i', frame_fmt, '-r', '30', video_path]
        subprocess.call(make_video)
        print('')
    
    
    def postprocess(self, trial_id, trial_metadata, imu2block, imu_settings):
        """
        []
        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [tuple(str)] trial_metadata:
          [0] participant id
          [1] birth month
          [2] birth year
          [3] gender
        [dict(str->str)] imu_settings:
        [dict(str->str)] imu2block:
        """
        
        imu_data, frame_timestamps = self.parseRawData(trial_id)
        
        self.writeImuSettings(trial_id, imu_settings)
        self.writeImuData(trial_id, imu_data)
        self.writeFrameTimestamps(trial_id, frame_timestamps)
                
        self.updateMetaData(trial_id, trial_metadata, imu2block)
        
        self.makeVideo(trial_id)
    
    
    def updateMetaData(self, trial_id, trial_metadata=None, imu2block=None):
        """
        Append or revise a row in the metadata array, then save the new array.
        When this method is called with only one argument, it only updates the
        'has labels' field.
        
        Args:
        -----
        [int] trial_id:
        [tuple(int)] trial_metadata:
        [dict(str->str)] imu2block:
        """
        
        print(imu2block)
        
        label_path = os.path.join(self.paths['labels'], str(trial_id) + '.csv')
        has_labels = int(os.path.exists(label_path))

        # Append a new row to the metadata array if the trial ID is one we
        # haven't seen before
        if self.meta_data.shape[0] <= trial_id:
            assert(not trial_metadata is None and not imu2block is None)
            block_mappings = tuple(imu2block[imu_id] for imu_id in self.imu_ids)
            meta_data = np.zeros(trial_id + 1, dtype=self.meta_data.dtype)
            meta_data[self.meta_data['trial id']] = self.meta_data
            meta_data[trial_id] = (trial_id,) + trial_metadata + (has_labels,) \
                                + block_mappings  
            self.meta_data = meta_data
        # If we've seen the trial ID before, we're revising a row
        else:
            self.meta_data[trial_id]['has labels'] = has_labels
            if not trial_metadata is None and not imu2block is None:
                block_mappings = tuple(imu2block[imu_id] for imu_id in self.imu_ids)
                self.meta_data[trial_id] = (trial_id,) + trial_metadata \
                                         + (has_labels,) + block_mappings
        
        self.writeMetaData()
    
    
    def makeImuFigs(self, trial_id):
        """
        Save plots of IMU acceleration, angular velocity, magnetic field
        (With annotations in the background, if they exist)
        
        Args:
        -----
        [int] trial_id:
        """
        
        # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
        imus = self.readImuData(trial_id)
        rgb_timestamps = self.readFrameTimestamps(trial_id)
        labels = self.readLabels(trial_id)
        
        # Define the output directory (for saving figures) and create it if it
        # doesn't already exist
        imu_fig_path = os.path.join(self.paths['figures'], 'imu', str(trial_id))
        if not os.path.exists(imu_fig_path):
            os.makedirs(imu_fig_path)
        
        fig_txt = [('Acceleration', 'a', 'g \/ \mathrm{m/s}^2'),
                   ('Angular velocity', '\omega', '\pi \/ \mathrm{rad/s}'),
                   ('Magnetic field', 'B', '\mathrm{mT}')]
        fn_abbrevs = ['accel', 'ang-vel', 'mag']
        
        norm_text = ('Square L2 norm', '\| \cdot \|', ' ')
        
        # Convert readings to natural units and plot, for each IMU selected
        for name, imu in imus.items():
            
            # Check the error column for bad samples
            bad_data = imu[:,1] == 1
            imu = imu[np.logical_not(bad_data),:]
            
            actions, imu_bounds = imuActionBounds(labels, rgb_timestamps[:,0], imu[:,0])
            
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
    
    """
    def makeCorrelationFigs(self, trial_id, devices):
        "
        Calculate pointwise cosine similarity for all pairs of devices in imus.
        
        Args:
        -----
        [int] trial_id:
        [list(str)] devices:
        "
        
        NUM_LABELS = 7
        
        # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
        imus = self.loadImuData(trial_id, devices)
        rgb_timestamps = self.loadRgbFrameTimestamps(trial_id)    
        labels = self.loadLabels(trial_id)
        
        # Define the output directory (for saving figures) and create it if it
        # doesn't already exist
        corr_fig_path = os.path.join('output', 'figures', 'imu-corrs', str(trial_id))
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
                    t_imu1 = imu1[np.logical_not(bad_data),0]
                    
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
        """
    
    
    def makeStateVisualizations(self, trial_id):
        """
        Write a series of graph images representing the block state sequence.
        
        Args:
        -----
        [int] trial_id:
        """
        
        labels = self.readLabels(trial_id)
        states = parseActions(labels)
        
        # Make a directory for the block state figures in this trial
        trial_dir = os.path.join(self.paths['figures'], 'block-states',
                                 str(trial_id))
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        
        for i, state in enumerate(states):
            state.render(filename=str(i), directory=trial_dir)
    
    
    def getRgbFrameFns(self, trial_id):
        """
        Return a sorted list of the RGB frame filenames for the specified
        trial.
        
        Args:
        -----
        [int] trial_id:
        
        Returns:
        --------
        [list(str)] rgb_frame_fns:
        """
        
        # Get full path names for RGB frames in this trial
        rgb_dir = os.path.join(self.paths['video-frames'], '{}-rgb'.format(trial_id))
        if not os.path.exists(rgb_dir):
            return []
        
        pattern = os.path.join(rgb_dir, '*.png')
        rgb_frame_fns = sorted(glob.glob(pattern))
        return rgb_frame_fns
        
        
    def makeVideoLabels(self, trial_id):
        """
        Interactive UI for labeling the actions in a trial based on video
        frames.
        
        Args:
        -----
        [int] trial_id:
        """
        
        if self.meta_data['has_labels'][trial_id]:
            fmtstr = 'Trial {}: Label file already exists. Delete it to re-label.'
            self.logger.warn(fmtstr.format(trial_id))
            return
        
        # Enable python 2/3 compatibility
        if sys.version_info[0] == 2:
            input = raw_input
        
        blocks = ('(none)', 'red square', 'yellow square', 'green square',
                  'blue square', 'red rectangle', 'yellow rectangle',
                  'green rectangle', 'blue rectangle')
        actions = ('(inactive)', 'place above', 'place adjacent', 'rotate',
                      'translate', 'remove', 'pick up (no placement)')
        
        # Get full path names for RGB frames in this trial
        rgb_frame_fns = self.getRgbFrameFns(trial_id)
        num_frames = len(rgb_frame_fns)
        
        # Set up image window
        cv2.namedWindow('Segmentation')
        
        label_fn = os.path.join(self.paths['labels'], '{}.csv'.format(trial_id))
        with open(label_fn, 'w') as label_file:
            labelwriter = csv.writer(label_file)
        
            action_started = False
            frame_idx = 0
            while True:
                
                # Open the current frame and display it
                frame_path = rgb_frame_fns[frame_idx]
                frame = cv2.imread(frame_path)
                cv2.imshow('Segmentation', frame)
                    
                # Wait until we read some input from the user
                user_in = cv2.waitKey(0)
                    
                # Act on the input if it's one of the recognized characters
                if action_started and user_in == ord('e'):
                    action_started = False
                    end_idx = frame_idx
                    
                    # Prompt user to choose an action
                    print('')
                    print('ACTIONS')
                    for i, a in enumerate(actions):
                        print('  {}: {}'.format(i, a))
                    action = int(input('Select the action: '))
                    
                    # Prompt user to choose object and target blocks
                    print('')
                    print('BLOCKS')
                    for i, b in enumerate(blocks):
                        print('  {}: {}'.format(i, b))
                    objects_str = input('Select the object block(s): ')
                    if len(objects_str.split()) > 1:
                        objects = (int(x) for x in objects_str.split())
                        target = 0
                        
                        # Write label and print to console
                        print('')
                        for obj in objects:
                            row = (start_idx, end_idx, action, obj, target, '()', '()')
                            labelwriter.writerow(row)
                            print((start_idx, end_idx, actions[action], blocks[obj],
                                   blocks[target]))
                    else:
                        obj = objects_str.split()[0]
                        targets_str = input('Select the target blocks(s): ')
                        targets = (int(x) for x in targets_str.split())
                    
                        # Write label and print to console
                        print('')
                        for target in targets:
                            row = (start_idx, end_idx, action, obj, target, '()', '()')
                            labelwriter.writerow(row)
                            print((start_idx, end_idx, actions[action], blocks[obj],
                                   blocks[target]))
                    
                    start_idx = -1
                    end_idx = -1
                elif user_in == ord('s'):
                    start_idx = frame_idx
                    action_started = True
                elif user_in == ord('j'):
                    if frame_idx - 1 > 0:
                        frame_idx -= 1
                    else:
                        frame_idx = 0
                        print('End of video')
                elif user_in == ord('k'):
                    if frame_idx + 1 < num_frames - 1:
                        frame_idx += 1
                    else:
                        frame_idx = num_frames - 1
                        print('End of video')
                elif user_in == ord('u'):
                    if frame_idx - 20 > 0:
                        frame_idx -= 20
                    else:
                        frame_idx = 0
                        print('End of video')
                elif user_in == ord('i'):
                    if frame_idx + 20 < num_frames - 1:
                        frame_idx += 20
                    else:
                        frame_idx = num_frames - 1
                        print('End of video')
                elif user_in == ord('q'):   # Exit
                    break
            
        self.meta_data['has_labels'][trial_id] = 1    
        self.writeMetaData()


if __name__ == '__main__':
    c = DuploCorpus()
        