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
import glob

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
        self.imu_ids = ('095D', '0949', '090F', '08F1')
        self.nickname2id = {'A':'095D', 'B':'0949', 'C':'090F', 'D':'08F1'}
        self.image_types = ('rgb', 'depth')
        
        self.initTypestructs()
        
        # Load metadata array
        self.meta_data = self.readMetaData()
    
    
    def initTypestructs(self):
        """
        Set up typestruct variables for saving and loading data
        """
        
        self.metadata_types = [('trial id', 'i4'), ('participant id', 'U10'), 
                               ('birth month', 'U3'), ('birth year', 'U4'),
                               ('gender', 'U15'), ('task id', 'i4')]        \
                            + [(name, 'U10') for name in self.imu_ids]        \
                            + [('has labels', 'i4')] 
        
        # 'studs' are length-18 strings because 18 is an upper bound on the
        # description length for this attribute: max 8 studs + 8 spaces + 2 parens
        self.label_types = [('start', 'i'), ('end', 'i'), ('action', 'i'),
                            ('object', 'i'), ('target', 'i'),
                            ('obj_studs', 'S25'), ('tgt_studs', 'S25')]
    
    
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
        meta_data:  numpy array
          Numpy structured array with the following fields
          trial id:  int
            This is just the entry's row index. All I/O operations take this
            id as an argument
          participant id:  str
            Anonymized participant identifier
          birth month:  str
            Participant birth month
          birth year:  str
            Participant birth year
          gender:  str
            Participant gender
          task id:  int
            Integer ID of the block construction task attempted by the
            participant
          08F1:  str
            Color of the rectangular block housing IMU 08F1
          095D:  str
            Color of the rectangular block housing IMU 095D
          090F:  str
            Color of the rectangular block housing IMU 090F
          0949:  str
            Color of the rectangular block housing IMU 0949
          has_labels:  int
            0 if human-annotated labels file does not exist, 1 if it does exist
        """

        meta_data = []
        
        # Start a new, empty metadata array if a file doesn't already exist
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        if not os.path.exists(fn):
            return np.array(meta_data, dtype=self.metadata_types)
        
        # Read contents of the metadata file line-by-line
        with open(fn, 'r') as metafile:
            metareader = csv.reader(metafile)
            next(metareader)   # Skip header
            for row in metareader:
                meta_data.append(tuple(row))
        
        return np.array(meta_data, dtype=self.metadata_types)
    
    
    def writeMetaData(self):
        """
        Write this object's metadata array to a file names meta-data.csv in the
        data directory.
        """
        
        # Write contents of the metadata array to file line-by-line
        fn = os.path.join(self.paths['data'], 'meta-data.csv')
        with open(fn, 'w') as metafile:
            metawriter = csv.writer(metafile)
            metawriter.writerow(self.meta_data.dtype.names)
            for row in self.meta_data:
                metawriter.writerow(row)
    
    
    def updateMetaData(self, trial_id, trial_metadata=None, imu2block=None):
        """
        Append or revise a row in the metadata array, then save the new array.
        When this method is called with only one argument, it only updates the
        'has labels' field.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        trial_metadata:  tuple of int
          (See readMetaData for descriptions of this tuple's entries)
          0 -- participant id
          1 -- birth month
          2 -- birth year
          3 -- gender
          4 -- block task ID
        imu2block:  dict of str -> str
          Dictionary mapping IMU names to the blocks housing them. This dict
          MUST contain an entry for every IMU in self.imu_ids.
        """
                
        label_path = os.path.join(self.paths['labels'], str(trial_id) + '.csv')
        has_labels = int(os.path.exists(label_path))

        # Append a new row to the metadata array if the trial ID is one we
        # haven't seen before
        if self.meta_data.shape[0] <= trial_id:
            assert(not trial_metadata is None and not imu2block is None)
            block_mappings = tuple(imu2block[imu_id] for imu_id in self.imu_ids)
            meta_data = np.zeros(trial_id + 1, dtype=self.meta_data.dtype)
            meta_data[self.meta_data['trial id']] = self.meta_data
            meta_data[trial_id] = (trial_id,) + trial_metadata + block_mappings \
                                + (has_labels,)                        
            self.meta_data = meta_data
        # If we've seen the trial ID before, we're revising a row
        else:
            self.meta_data[trial_id]['has labels'] = has_labels
            if not trial_metadata is None and not imu2block is None:
                block_mappings = tuple(imu2block[imu_id] for imu_id in self.imu_ids)
                self.meta_data[trial_id] = (trial_id,) + trial_metadata       \
                                         + block_mappings + (has_labels,)
        
        self.writeMetaData()
    
    
    def writeImuSettings(self, trial_id, imu_settings):
        """        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        dev_settings:  numpy array
          Numpy structured array. Each row contains the settings of one IMU.
        """
        
        # Write settings array to CSV line-by-line
        fn = os.path.join(self.paths['imu-settings'], '{}.csv'.format(trial_id))
        with open(fn, 'w') as settings_file:
            settings_writer = csv.writer(settings_file)
            settings_writer.writerow(imu_settings.dtype.names)
            for row in imu_settings:
                settings_writer.writerow(row)
    
    
    def readImuData(self, trial_id):
        """
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        
        Returns:
        --------
        imu_data:  dict of str -> numpy array
          Dictionary mapping the name of each IMU that was used in this trial
          to a numpy array representing a timeseries of data samples. Columns
          of this array are
          0 ----- Global timestamp. PC time when sample was received, in seconds
          1 ----- Error flag. 0 if sample is usable (no error), 1 if an
                  error occurred.
          2 ----- Sample index (starts at 1)
          3 ----- IMU timestamp. IMU's local time when sample was taken, in
                  seconds
          4:6 --- Acceleration meaured with respect to the IMU frame in m/sec^2
                  (XYZ order)
          7:9 --- Angular velocity meaured with respect to the IMU frame in
                  degrees / sec (XYZ order)
          10:12 - Magnetic field meaured with respect to the IMU frame in 
                  milliGauss (XYZ order)
          13 ---- Battery voltage in milliVolts (only sampled about once or
                  twice per second)
          14 ---- Temperature (only sampled about once or twice per second)
          15 ---- Barometric pressure (only sampled about once or twice per
                  second)
        """
        
        imu_data = {}
        for imu_id in self.imu_ids:
            
            # IMUs that weren't used during the trial won't have any data to
            # read (obviously)
            if self.meta_data[imu_id][trial_id] == 'UNUSED':
                continue
            
            fn = '{}-{}.csv'.format(trial_id, imu_id)
            path = os.path.join(self.paths['imu-samples'], fn)
            # Skip the first line because it's just the column names
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            
            # FIXME: convert intelligently
            data[:,3] = data[:,3] / 65536.0   # Convert sample timestamp to seconds
            data[:,4:7] = data[:,4:7] / 4096.0    # Convert acceleration samples to g
            data[:,7:10] = data[:,7:10] * 0.07  # Convert angular velocity samples to deg/s
            data[:,10:13] = data[:,10:13] * 1e-3  # Convert magnetic field to units of mGauss
            data[:,12] = - data[:,12]   # Field is measured along -z instead of +z
            
            # TODO: Convert temp and pressure to metric units
            
            imu_data[imu_id] = data
        
        return imu_data


    def writeImuData(self, trial_id, imu_data):
        """
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        imu_data:  dict of str -> numpy array
          (Same as output of readImuData)
        """
        
        for name, data in imu_data.items():
            fn = '{}-{}.csv'.format(trial_id, name)
            path = os.path.join(self.paths['imu-samples'], fn)
            sample_len = data.shape[1]
            fmtstr = ['%15f'] + (sample_len - 1) * ['%i']
            col_names = 'global timestamp,error,sample index,imu timestamp,'\
                        'ax,ay,az,gx,gy,gz,mx,my,-mz,voltage,temp,pressure'
            np.savetxt(path, data, delimiter=',', fmt=fmtstr, header=col_names,
                       comments='')
    
    
    def readFrameTimestamps(self, trial_id):
        """
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
                
        Returns:
        --------
        frame_timestamps:  dict of str -> numpy array
          Dictionary mapping the name of each camera that was used in this
          trial to a numpy array representing a timeseries of data samples.
          Columns of this array are
          0 -- Global timestamp. PC time when sample was received, in seconds
          1 -- Error flag. 0 if sample is usable (no error), 1 if an error
               occurred.
          2 -- Sample index (this is also the name of the corresponding video
               frame captured by the camera) (starts at 0)
        """
        
        for image_type in self.image_types:
            fn = '{}-{}.csv'.format(trial_id, image_type)
            path = os.path.join(self.paths['frame-timestamps'], fn)
            # Skip the first line because it's just the column names
            frame_timestamps = np.loadtxt(path, delimiter=',', skiprows=1)
        
        return frame_timestamps
        
    
    def writeFrameTimestamps(self, trial_id, frame_timestamps):
        """
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        img_timestamps:  dict of str -> numpy array
          (Same as output of readFrameTimestamps)
        """
        
        for name, timestamps in frame_timestamps.items():
            fn = '{}-{}.csv'.format(trial_id, name)
            path = os.path.join(self.paths['frame-timestamps'], fn)
            sample_len = timestamps.shape[1]
            fmtstr = ['%15f'] + (sample_len - 1) * ['%i']
            col_names = 'global timestamp,error,sample index'
            np.savetxt(path, timestamps, delimiter=',', fmt=fmtstr,
                       header=col_names, comments='')
    
    
    def readLabels(self, trial_id, annotator_id):
        """
        Try to load annotations from file for the given trial. If
        there are no labels, warn the user and return an empty list.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        annotator_id:  str
          Annotator identifier. This is usually the annotator's name or
          something similar.
        
        Returns:
        --------
        labels:  numpy array
          Structured array with the following fields
          start:  int
            Video frame index for start of event
          end:  int
            Video frame index for end of event
          action:  int
            Integer event ID (0-5). See labels file for description
          object:  int
            Integer ID for object of event. Ex: 1 in 'place 1 on
            4'. Not all events have objects; when object N/A ID is 0
          target:  int
            Integer ID for target of event. Ex: 4 in 'place 1 on
            4'. Not all events have targets; when target is N/A ID is 0
          obj_studs:  str
            Adjacent studs are recorded for placement events. The
            i-th entry in this list is adjacent to the i-th entry in tgt_studs.
          tgt_studs:  str
            Same as above. The i-th entry in this list is adjacent
            to the i-th entry in obj_studs.
        """
        
        fn = os.path.join(self.paths['labels'], '{}-{}.csv'.format(trial_id, annotator_id))
        
        if not os.path.exists(fn):
            self.logger.warn('Trial {}: no labels found'.format(trial_id))
            return []
        
        # the first row is the column names, so skip it
        labels = np.loadtxt(fn, delimiter=',', dtype=self.label_types,
                            skiprows=1)
        return labels
    
    
    def writeLabels(self, trial_id, annotator_id, labels):
        """
        Write labels to file for the given trial.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        annotator_id: str
          String identifying the person who annotated this set of labels.
        labels:  list of tuple
          Labels to be written. Each entry (tuple) in the
          list represents a single action. Entries are as follows
          0 -- Video frame index for start of event
          1 -- Video frame index for end of event
          2 -- Integer event ID (0-5). See labels file for description
          3 -- Integer ID for object of event. Ex: 1 in 'place 1 on
               4'. Not all events have objects; when object N/A ID is 0
          4 -- Integer ID for target of event. Ex: 4 in 'place 1 on
               4'. Not all events have targets; when target is N/A ID is 0
          5 -- Adjacent studs are recorded for placement events. The
               i-th entry in this list is adjacent to the i-th entry in tgt_studs.
          6 -- Same as above. The i-th entry in this list is adjacent
               to the i-th entry in obj_studs.
        """
        
        # Convert labels to numpy structured array
        labels = np.array(labels, dtype=self.label_types)
        
        fn = '{}-{}.csv'.format(trial_id, annotator_id)
        path = os.path.join(self.paths['labels'], fn)
        with open(path, 'w') as labels_file:
            labels_writer = csv.writer(labels_file)
            labels_writer.writerow(labels.dtype.names)
            for row in labels:
                labels_writer.writerow(row)
    
    
    def labelFileExists(self, trial_id, annotator_id):
        """
        Return True if a label file matching the given trial and annotator
        exists. This is important for making sure labels don't get overwritten.
        """
        
        fn = '{}-{}.csv'.format(trial_id, annotator_id)
        path = os.path.join(self.paths['labels'], fn)
        return os.path.isfile(path)
    
    
    def writeNotes(self, trial_id, annotator_id, notes):
        """
        Write notes to file for the given trial.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        annotator_id: str
          String identifying the person who annotated this set of notes.
        notes:  list of tuple
          Notes to be written. Entries are as follows:
          0 -- video frame when note was made
          1 -- string representing the note
        """
        
        fn = '{}-{}-notes.csv'.format(trial_id, annotator_id)
        path = os.path.join(self.paths['labels'], fn)
        with open(path, 'w') as notes_file:
            notes_writer = csv.writer(notes_file)
            for row in notes:
                notes_writer.writerow(row)
    
    
    def parseRawData(self, trial_id):
        """
        Read raw data from file and convert to numpy arrays
    
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
          
        Returns:
        --------
        [TODO]
        """
        
        imu_data = self.parseImuData(trial_id)
        
        frame_timestamps = self.parseVideoData(trial_id)
        
        return (imu_data, frame_timestamps)
    
    
    def parseImuData(self, trial_id):
        """
        Read raw data from file and convert to numpy arrays
    
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
          
        Returns:
        --------
        [TODO]
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
        
        return imu_data
    
    
    def parseVideoData(self, trial_id):
        """
        Read raw data from file and convert to numpy arrays
    
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
          
        Returns:
        --------
        [TODO]
        """
        
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
        
        return frame_timestamps
    
    
    def postprocess(self, trial_id, trial_metadata, imu2block, imu_settings):
        """
        This is a wrapper script that parses the raw IMU and video frame
        timestamp data into more organized formats and updates the metadata
        array to record that the trial occurred.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the
          corpus metadata array.
        trial_metadata:  tuple of str
          (See updateMetaData)
          0 -- participant id
          1 -- birth month
          2 -- birth year
          3 -- gender
          4 -- block task ID
        imu2block:  dict of str -> str
          (See updateMetaData)
        imu_settings:  dict of str -> str
          (See writeImuSettings)
        """
        
        imu_data, frame_timestamps = self.parseRawData(trial_id)
        
        self.writeImuSettings(trial_id, imu_settings)
        self.writeImuData(trial_id, imu_data)
        self.writeFrameTimestamps(trial_id, frame_timestamps)
                
        self.updateMetaData(trial_id, trial_metadata, imu2block)
            
    
    def makeImuFigs(self, trial_id):
        """
        Save plots of IMU acceleration, angular velocity, magnetic field
        (With annotations in the background, if they exist)
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        """
        
        # FIXME: With the way I'm currently drawing background annotations,
        #   this method will break with simultaneous events
        
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
    
    
    def makeStateVisualizations(self, trial_id):
        """
        Write a series of graph images representing the block state sequence.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
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
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        
        Returns:
        --------
        rgb_frame_fns:  list of str
          Full path to each RGB frame, sorted in alphanumeric order
        """
        
        # Get full path names for RGB frames in this trial
        rgb_dir = os.path.join(self.paths['video-frames'], '{}-rgb'.format(trial_id))
        if not os.path.exists(rgb_dir):
            return []
        
        pattern = os.path.join(rgb_dir, '*.png')
        rgb_frame_fns = sorted(glob.glob(pattern))
        return rgb_frame_fns
    
    
    def getDepthFrameFns(self, trial_id):
        """
        Return a sorted list of the depth frame filenames for the specified
        trial.
        
        Args:
        -----
        trial_id:  int
          Trial identifier. This is the trial's index in the corpus metadata
          array.
        
        Returns:
        --------
        depth_frame_fns:  list of str
          Full path to each depth frame, sorted in alphanumeric order
        """
        
        # Get full path names for RGB frames in this trial
        rgb_dir = os.path.join(self.paths['video-frames'], '{}-depth'.format(trial_id))
        if not os.path.exists(rgb_dir):
            return []
        
        pattern = os.path.join(rgb_dir, '*.png')
        depth_frame_fns = sorted(glob.glob(pattern))
        return depth_frame_fns


if __name__ == '__main__':
    
    # Ignore the code below if you aren't me (the author)
    
    c = DuploCorpus()
    c.makeImuFigs(24)
    """
    sample_rate = 15    # Hz
    end_idx = 45 * sample_rate  # No motion for the first ~45-50 seconds
    imu_data = c.readImuData(13)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for name, data in imu_data.items():
        a_stationary = data[:end_idx, 4:7]
        mu_a = a_stationary.mean(axis=0)
        theta_xz = np.arctan2(mu_a[0], mu_a[2])
        theta_yz = np.arctan2(mu_a[1], mu_a[2])
        print()
        print(mu_a)
        fmtstr = 'XZ angle: {:.5f}  |  YZ angle: {:.5f}'
        print(fmtstr.format(np.rad2deg(theta_xz), np.rad2deg(theta_yz)))
        line = np.vstack((np.zeros(3), mu_a))
        ax.plot(line[:,0], line[:,1], line[:,2])
        ax.scatter(mu_a[0], mu_a[1], mu_a[2])
        """
