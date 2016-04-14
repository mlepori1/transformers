"""
streamdata.py
  Connect to four WAX9 IMUs and a PrimeSense camera, stream data, write to CSV

AUTHOR
  Jonathan D. Jones
"""

import numpy as np
import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
from primesense import openni2
import cv2
import csv
import os
import sys
import time
import subprocess
from itertools import cycle
import struct

from libwax9 import *
from postprocess import *


def streamVideo(devices, q, die, paths):
    """
    Stream data from camera until die is set

    Args:
    -----
    [dict(str->cv stream)] devices:
    [mp queue] q: Multiprocessing queue, for collecting data
    [mp event] die: Multiprocessing event, kill signal
    [dict(str->str)] paths: Path (full or relative) to image output directory
    """
        
    # This allows us to cycle through device names -> sockets in the main loop
    dev_names = cycle(devices.keys())
    
    frame_index = {name: 0 for name in devices.keys()}
    
    for stream in devices.values():
        stream.start()    
    
    dev_name = dev_names.next()
    stream = devices[dev_name]
    index = frame_index[dev_name]
    path = paths[dev_name]
    while not die.is_set():
        """
        # Read depth frame data, convert to image matrix, write to file,
        # record frame timestamp
        frametime = time.time()
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_array = np.ndarray((depth_frame.height, depth_frame.width),
                                 dtype=np.uint16, buffer=depth_data)
        filename = "depth_" + str(i) + ".png"
        cv2.imwrite(os.path.join(dirname, filename), depth_array)
        q.put((i, frametime, "IMG_DEPTH"))
        """
        
        # Read frame data, convert to image matrix, write to file,
        # record frame timestamp
        frametime = time.time()
        frame = stream.read_frame()
        data = frame.get_buffer_as_uint8()
        # FIXME: depth streams should be parsed differently
        img_array = np.ndarray((frame.height, 3*frame.width),
                               dtype=np.uint8, buffer=data)
        img_array = np.dstack((img_array[:,2::3], img_array[:,1::3], img_array[:,0::3]))
        filename = '{:06d}.png'.format(index)
        cv2.imwrite(os.path.join(path, filename), img_array)
        q.put((frametime, index, dev_name))

        frame_index[dev_name] += 1
        
        # Cycle to the next device
        dev_name = dev_names.next()
        stream = devices[dev_name]
        index = frame_index[dev_name]
        path = paths[dev_name]


def streamImu(devices, q, die):
    """
    Stream data from WAX9 devices until die is set

    Args:
    -----
    [dict(str->socket)] devices: Dict w/ device names as keys and their
      sockets as values
    [mp queue] q: Multiprocessing queue, for collecting data
    [mp event] die: Multiprocessing event, kill signal
    """

    # This allows us to cycle through device names -> sockets in the main loop
    dev_names = cycle(devices.keys())

    # Tell the device to start streaming
    for socket in devices.values():
        socket.sendall('stream\r\n')

    # Read data sequentially from sensors until told to terminate
    error = 0
    prev_frame = ''
    data = [0] * 11
    dev_id = dev_names.next()
    socket = devices[dev_id]
    while True:
        
        # Read at most 28 bytes over RFCOMM. If the packet is length-36, it
        # will be read in two passes.
        frame = prev_frame + socket.recv(28)
        cur_time = time.time()
        
        # The WAX9 device transmits SLIP-encoded data: packets begin and end
        # with 0xC0
        packet = []
        if not frame[0] == '\xc0':
            # We missed the beginning of this packet. There's nothing we can
            # do to fix it, so throw it out
            # FIXME: Make sure we don't throw out the beginning of another
            # packet that may be included in this frame?
            prev_frame = ''
            error = 1
        elif len(frame) < 28:
            # Not long enough to be a packet yet, so keep appending to frame
            prev_frame = frame
        elif frame[2] == '\x01' and frame[27] == '\xc0':
            # The first 28 bytes are a standard WAX9 packet. Process it and
            # start appending to what's left (if anything)
            packet = frame[:28]
            prev_frame = frame[28:] if len(frame) > 28 else ''
            error = 0        
        elif len(frame) < 36:
            # Shorter than 36 bytes with the first 28 NOT a standard packet:
            # this might still be a long packet, so keep appending to frame
            prev_frame = frame
        elif frame[2] == '\x01' and frame[28] == '\xc0':
            # Anomalous length-29 packet
            # extra byte might come from a SLIP escape character
            packet = frame[:29]
            prev_frame = frame[29:] if len(frame) > 29 else ''
            error = 1
        elif frame[2] == '\x02' and frame[35] == '\xc0':
            # The first 36 bytes are a long WAX9 packet. Process it and
            # start appending to what's left (if anything)
            packet = frame[:36]
            prev_frame = frame[36:] if len(frame) > 36 else ''
            error = 0
        elif len(frame) > 36 and frame[2] == '\x02' and frame[36] == '\xc0':
            # Anomalous length-37 packet
            # extra byte might come from a SLIP escape character
            packet = frame[:37]
            prev_frame = frame[37:] if len(frame) > 37 else ''
            error = 1
        else:
            # We've read more than 36 bytes and haven't found a standard
            # packet or a long packet, so throw out this frame and start over
            prev_frame = ''
            error = 1
            
        if packet or error:
            if len(packet) == 28:
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                #print('01 | {} | {}'.format(len(frame), frame.encode('hex')))
                data = [cur_time, error]
                data += list(struct.unpack('<BBBhIhhhhhhhhhB', packet))[3:27]
                data.append(dev_id)
            elif len(packet) == 36:
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                #print('02 | {} | {}'.format(len(frame), frame.encode('hex')))
                data = [cur_time, error]
                data += list(struct.unpack('<BBBhIhhhhhhhhhhhIB', packet))[3:27]
                data.append(dev_id)
            elif error:
                # Record an error
                # FIXME: Try to write the previous sample
                print('ERR | {} | {} | {}'.format(dev_id, len(frame), frame.encode('hex')))
                print('    | {} | {} | {}'.format(dev_id, len(prev_frame), prev_frame.encode('hex')))
                data = [0] * 26
                data[1] = error
            
            # Queue the packet to be written
            q.put(data)
            
            # Cycle to the next device
            dev_id = dev_names.next()
            socket = devices[dev_id]
        
        # Terminate when main tells us to
        if die.is_set():
            print('Dying...')
            # Tell the device to stop streaming
            for socket in devices.values():
                socket.sendall('\r\n')
            break


def write(path, q, die):
    """
    Write data on queue to file until a kill signal is received. Once the kill
    signal is received, write what's left on the queue, then terminate

    Args:
    -----
    [str] path: Path (full or relative) to output file
    [mp queue] q: Multiprocessing queue, for collecting data
    [mp event] die: Multiprocessing event, kill signal
    """

    with open(path, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        received_die = die.is_set()
        while not q.empty() or not received_die:
            # Keep checking for a kill message until we get one
            if not received_die:
                received_die = die.is_set()
            # Write data to file
            if not q.empty():
                line = q.get()
                csvwriter.writerow(line)


def raw2npArray(path, imu_dev_names, img_dev_names):
    """
    Convert raw data to numpy array

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

    imu_data = []
    img_data = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            sample_idx = int(row[0])
            timestamp = float(row[-2])

            if row[-1] in img_devs:
                # Append rows of zeros to the data matrix until the last row
                # is the sample index. Then write the current sample to its
                # place in the sample index.
                img_dev_idx = img_name2idx[row[-1]]
                for i in range(sample_idx - (len(img_data) - 1)):
                    img_data.append([0.0] * num_img_devs)
                img_data[sample_idx][img_dev_idx] = timestamp
            elif row[-1] in imu_devs:
                sample = [int(x) for x in row[:-1]]
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
    return (np.array(imu_data), np.array(img_data), sample_len)


def saveSettings(path, dev_settings):
    """
    Record settings for each IMU device.
    
    Args:
    -----
    [str] path: Path (full or relative) to output file
    [dict(str->str)] dev_settings: Dict mapping device IDs to recorded
      settings 
    """
    
    path = os.path.join('data', 'dev-settings', '{}.txt'.format(init_time))
    with open(path, 'wb') as file:
        for settings in dev_settings.values():
            file.write(settings)
            file.write('\n')


def printPercentDropped(imu_data, dev_names, sample_len):
    """
    Calculate the proportion of missing or unusable data samples in the
    provided data array; print to console

    Args:
    -----
    [np array] imu_data: Size n-by-m*d, where n is the number of IMU
      samples, m is the number of IMU devices, and d is the length of one IMU
      sample.
    [list(str)] dev_names: List of IMU ID strings; define m as the number of
      items in this list
    [int] sample_len: Length of one IMU sample (d)

    Returns:
    --------
    [float] percent_dropped: Proportion of missing data samples over all
      devices, expressed as a percentage
    """

    # These are aggregate drop/sample counts
    total_dropped = 0.0
    total_samples = 0.0

    # For each device, count the number of timestamps near 0. These samples
    # were corrupted (dropped) during data transmission
    for i, name in enumerate(dev_names):
        # Timestamp is the first datum in a sample
        timestamp_idx = sample_len * i
        num_dropped = np.sum(np.less_equal(imu_data[:,timestamp_idx], 1.0))
        num_samples = imu_data[:,timestamp_idx].shape[0]

        percent_dropped = float(num_dropped) / float(num_samples) * 100
        fmtstr = '{}: {:0.1f}% of {} samples dropped'
        print(fmtstr.format(name, percent_dropped, num_samples))

        total_dropped += num_dropped
        total_samples += num_samples

    # Percent of samples dropped over all devices
    percent_dropped = total_dropped / total_samples * 100
    fmtstr = 'TOTAL: {:0.1f}% of {} samples dropped'
    print(fmtstr.format(percent_dropped, int(total_samples)))

    return percent_dropped


if __name__ == "__main__":
    
    # raw_input() in python 2.x is input() in python 3.x
    # Below is for compatibility with python 2
    if sys.version_info[0] == 2:
        input = raw_input
    
    # Define pathnames and filenames for the data from this trial
    init_time = str(int(time.time()))   # Used as a trial identifier
    settings_path = os.path.join('data', 'dev-settings')
    raw_path = os.path.join('data', 'raw')
    imu_path = os.path.join('data', 'imu')
    rgb_path = os.path.join('data', 'rgb')
    rgb_trial_path = os.path.join(rgb_path, init_time)
    rgb_timestamp_path = os.path.join(rgb_trial_path, 'frame-timestamps.csv')
    raw_file_path = os.path.join(raw_path, init_time + '.csv')
    imu_file_path = os.path.join(imu_path, init_time + '.csv')
    settings_file_path = os.path.join(settings_path, init_time + '.txt')

    # Create data directories if they don't exist
    for path in [settings_path, raw_path, imu_path, rgb_trial_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Bluetooth MAC addresses of the IMUs we want to stream from
    addresses = ('00:17:E9:D7:08:F1',)
                 #'00:17:E9:D7:09:5D',
                 #'00:17:E9:D7:09:0F',
                 #'00:17:E9:D7:09:49')

    # Connect to devices and print settings
    imu_devs = {}
    imu_settings = {}
    for address in addresses:
        print('Connecting at {}...'.format(address))
        # TODO: check if name is empty string and retry if so
        socket, name = connect(address)
        print('Connected, device ID {}'.format(name))
        settings = getSettings(socket)
        print(settings)
        imu_devs[name] = socket
        imu_settings[name] = settings
    
    # Open video streams
    img_devs = {}
    img_paths = {}
    openni2.initialize()
    dev = openni2.Device.open_any()
    #img_devs['IMG-DEPTH'] = dev.create_depth_stream()
    img_devs['IMG-RGB'] = dev.create_color_stream()
    img_paths['IMG-RGB'] = rgb_trial_path

    # Start receiving and writing data
    q = SimpleQueue()
    die = mp.Event()
    processes = []
    processes.append(mp.Process(target=streamImu, args=(imu_devs, q, die)))
    #processes.append(mp.Process(target=streamVideo, args=(img_devs, q, die, img_paths)))
    processes.append(mp.Process(target=write, args=(raw_file_path, q, die)))
    for p in processes:
        p.start()

    # Wait for kill signal from user
    while True:
        user_in = input('Streaming... (press return to stop)\n')
        if not user_in:
            print('Killing processes...')
            die.set()
            break
    
    for p in processes:
        p.join()

    # Disconnect from devices
    for dev_name, socket in imu_devs.items():
        print('Disconnecting from {}'.format(dev_name))
        socket.close()
    
    # Stop streaming
    for dev_name, stream in img_devs.items():
        stream.stop()
    openni2.unload()
    
    """
    # Massage and save IMU, RGB frame data
    saveSettings(settings_file_path, dev_settings)
    imu_data, img_data, sample_len = raw2npArray(raw_file_path, imu_devs.keys(), img_devs.keys())
    fmtstr = len(devices) * (['%15f'] + (sample_len - 1) * ['%i'])
    np.savetxt(imu_file_path, imu_data, delimiter=',', fmt=fmtstr)
    np.savetxt(img_timestamp_path, img_data, delimiter=',', fmt='%15f')
    
    # Convert rgb frames to avi video
    # NOTE: This depends on the avconv utility for now
    print('')
    frame_fmt = os.path.join(rgb_trial_path, '%6d.png')
    video_path = os.path.join(rgb_path, init_time + '.avi')
    make_video = ['avconv', '-f', 'image2', '-i', frame_fmt, '-r', '30', video_path]
    subprocess.call(make_video)
    print('')
    
    percent_dropped = printPercentDropped(imu_data, imu_devs.keys(), sample_len)
    
    # Show IMU data (for validation)
    ids = [x[-4:] for x in imu_devs.keys()]  # Grab hex ID from WAX9 ID
    plotImuData(int(init_time), ids)
    """
    