# -*- coding: utf-8 -*-
"""
duplocorpus.py
  Wrapper class for block study data (IMU, video, labels, meta, etc)

AUTHOR
  Jonathan D. Jones
"""

import os
import logging
import csv
import subprocess

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from libduplo import *


class DuploCorpus:
    
    
    def __init__(self):
        self.initPaths()
        self.initLogger()
        
        # Load metadata array
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        self.metadata_typestruct = [('trial_id', 'U10'), ('child_id', 'U10'),
                                    ('has_labels', 'bool')]
        self.meta_data = np.loadtxt(fn, dtype=self.metadata_typestruct,
                                    delimiter=',')
        
    
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
        self.paths['src'] = os.path.split(__file__)[0]
        self.paths['root'] = os.path.dirname(self.paths['src'])
        self.paths['output'] = os.path.join(self.paths['root'], 'output')        
        self.paths['working'] = os.path.join(self.paths['root'], 'working')
        self.paths['data'] = os.path.join(self.paths['root'], 'data')
        self.paths['imu'] = os.path.join(self.paths['data'], 'imu')
        self.paths['imu-settings'] = os.path.join(self.paths['data'], 'imu-settings')
        self.paths['rgb'] = os.path.join(self.paths['data'], 'rgb')
        self.paths['raw'] = os.path.join(self.paths['data'], 'raw')
        self.paths['labels'] = os.path.join(self.paths['data'], 'labels')
        
        # Make directories that do not surely exist already
        for key in ['working', 'output']:
            if not os.path.exists(self.paths[key]):
                os.makedirs(self.paths[key])        
    
    
    def writeImuSettings(self, trial_id, dev_settings):
        """        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [dict(str->str)] dev_settings: Dict mapping device IDs to recorded
          settings 
        """
        
        fn = os.path.join(self.paths['imu-settings'], '{}.txt'.format(trial_id))
        with open(fn, 'wb') as settings_file:
            for settings in dev_settings.values():
                settings_file.write(settings)
                settings_file.write('\n')
    
    
    def readImuData(self, trial_id):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        
        Returns:
        --------
        [np array] imu_data:
        """
        
        fn = os.path.join(self.paths['imu'], '{}.csv'.format(trial_id))
        #fmtstr = num_imus * (['%15f'] + (sample_len - 1) * ['%i'])
        imu_data = np.loadtxt(fn, delimiter=',')
        
        return imu_data


    def writeImuData(self, trial_id, imu_data, num_imus, sample_len):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [np array] imu_data:
        """
        
        fn = os.path.join(self.paths['imu'], '{}.csv'.format(trial_id))
        fmtstr = num_imus * (['%15f'] + (sample_len - 1) * ['%i'])
        np.savetxt(fn, imu_data, delimiter=',', fmt=fmtstr)
    
    
    def readRgbTimestamps(self, trial_id):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
                
        Returns:
        --------
        [np array] rgb_timestamps:
        """
        
        fn = os.path.join(self.paths['rgb'], str(trial_id), 'frame-timestamps.csv')
        rgb_timestamps = np.loadtxt(fn, delimiter=',')
        
        return rgb_timestamps
        
    
    def writeRgbTimestamps(self, trial_id, rgb_timestamps):
        """
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        [np array] rgb_timestamps:
        """
        
        fn = os.path.join(self.paths['rgb'], str(trial_id), 'frame-timestamps.csv')
        np.savetxt(fn, rgb_timestamps, delimiter=',', fmt='%15f')
    
    
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
        typestruct = [('start', 'i'), ('end', 'i'), ('action', 'i'),
                      ('object', 'i'), ('target', 'i'), ('obj_studs', 'S18'),
                      ('tgt_studs', 'S18')]
        labels = np.loadtxt(fn, delimiter=',', dtype=typestruct)
        return labels
    
    
    def parseRawData(self, trial_id, imu_dev_names, img_dev_names):
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
        [np array] imu_data: Size n-by-p*d, where n is the number of IMU
          samples, p is the number of IMU devices, and d is the length of one IMU
          sample. Each holds p length-d IMU samples, concatenated in the order
          seen in imu_dev_names
        [np array] img_data: Size m-by-p. Contains the global timestamp for every video
          frame recorded
        [int] sample_len: Length of one IMU sample (d)
        """
        
        num_imu_devs = len(imu_dev_names)
        imu_name2idx = {imu_dev_names[i]: i for i in range(num_imu_devs)}
        
        num_img_devs = len(img_dev_names)
        img_name2idx = {img_dev_names[i]: i for i in range(num_img_devs)}
        
        path = os.path.join(self.paths['raw'], '{}.csv'.format(trial_id))
    
        imu_data = []
        img_timestamps = []
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                timestamp = float(row[0])
                dev_name = row[-1]
    
                if dev_name in img_dev_names:
                    sample_idx = int(row[1])
                    
                    # Append rows of zeros to the data matrix until the last row
                    # is the sample index. Then write the current sample to its
                    # place in the sample index.
                    img_dev_idx = img_name2idx[dev_name]
                    for i in range(sample_idx - (len(img_timestamps) - 1)):
                        img_timestamps.append([0.0] * num_img_devs)
                    img_timestamps[sample_idx][img_dev_idx] = timestamp
                elif dev_name in imu_dev_names:
                    sample_idx = int(row[2])
                    sample = [timestamp] + [int(x) for x in row[1:-1]]
                    sample_len = len(sample)
                    
                    # Append rows of zeros to the data matrix until the last row
                    # is the sample index. Then write the current sample to its
                    # place in the sample index.
                    imu_dev_idx = imu_name2idx[row[-1]]
                    for i in range(sample_idx - (len(imu_data) - 1)):
                        imu_data.append([[0.0] * sample_len] * num_imu_devs)
                    imu_data[sample_idx][imu_dev_idx] = sample
        
        # Flatten nested lists in each row of IMU data
        imu_data = [[datum for sample in row for datum in sample] for row in imu_data]
        return (np.array(imu_data), np.array(img_timestamps), sample_len)
    
    
    def makeRgbVideo(self, trial_id):
        """
        Convert RGB frames to avi video
        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        """
        
        # NOTE: This depends on the avconv utility for now
        print('')
        frame_fmt = os.path.join(self.paths['rgb'], str(trial_id), '%6d.png')
        video_path = os.path.join(self.paths['rgb'], '{}.avi'.format(trial_id))
        make_video = ['avconv', '-f', 'image2', '-i', frame_fmt, '-r', '30', video_path]
        subprocess.call(make_video)
        print('')
    
    
    def postprocess(self, child_id, trial_id, imu_devs, imu_settings, img_dev_name):
        """
        []
        
        Args:
        -----
        [int] trial_id: Trial identifier. This is the trial's index in the
          corpus metadata array.
        """
        
        imu_data, rgb_timestamps, sample_len = self.parseRawData(trial_id, imu_devs.keys(), (img_dev_name,))
        
        self.writeImuSettings(trial_id, imu_settings)
        self.writeImuData(trial_id, imu_data, len(imu_devs), sample_len)
        self.writeRgbTimestamps(trial_id, rgb_timestamps)
        #self.makeRgbVideo(trial_id)
        
        label_path = os.path.join(self.paths['labels'], str(trial_id) + '.csv')
        has_labels = os.path.exists(label_path)
        
        meta_row = np.array((trial_id, child_id, has_labels),
                            dtype=self.metadata_typestruct)
        self.meta_data = np.hstack((self.meta_data, meta_row))
        
        # Update metadata
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        np.savetxt(self.meta_data, fn, dtype=self.metadata_typestruct,
                   delimiter=',')
    
    
    def makeImuFigs(self, trial_id, devices):
        """
        Save plots of IMU acceleration, angular velocity, magnetic field
        (With annotations in the background, if they exist)
        
        Args:
        -----
        [int] trial_id:
        [list(str)] devices:
        """
        
        # Load IMU data, RGB frame timestamps, and labels (empty list if no labels)
        imus = self.readImuData(trial_id, devices)
        rgb_timestamps = self.readRgbFrameTimestamps(trial_id)
        labels = self.readLabels(trial_id)
        
        # Define the output directory (for saving figures) and create it if it
        # doesn't already exist
        imu_fig_path = os.path.join('output', 'figures', 'imu', str(trial_id))
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
    

    
    def makeCorrelationFigs(self, trial_id, devices):
        """
        Calculate pointwise cosine similarity for all pairs of devices in imus.
        
        Args:
        -----
        [int] trial_id:
        [list(str)] devices:
        
        Returns:
        --------
          (Nothing)
        """
        
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


if __name__ == '__main__':
    c = DuploCorpus()