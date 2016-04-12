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

from libwax9 import *
from postprocess import *


def streamVideo(path, q, die):
    """
    Stream data from camera until die is set

    Args:
    -----
    [str] path: Path (full or relative) to output file
    [mp queue] q: Multiprocessing queue, for collecting data
    [mp event] die: Multiprocessing event, kill signal
    """

    # Start streaming rgb video
    openni2.initialize()
    dev = openni2.Device.open_any()
    #depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    for stream in [color_stream]: #, depth_stream]:
        stream.start()

    i = 0
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

        # Read color frame data, convert to image matrix, write to file,
        # record frame timestamp
        frametime = time.time()
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color_array = np.ndarray((color_frame.height, 3*color_frame.width),
                                 dtype=np.uint8, buffer=color_data)
        color_array = np.dstack((color_array[:,2::3], color_array[:,1::3],
            color_array[:,0::3]))
        filename = 'rgb_{:06d}.png'.format(i)
        cv2.imwrite(os.path.join(path, filename), color_array)
        q.put((i, frametime, 'IMG_RGB'))

        i += 1

    # Stop streaming
    #depth_stream.stop()
    color_stream.stop()
    openni2.unload()


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
    prev_frame = ''
    dev_id = dev_names.next()
    socket = devices[dev_id]
    while True:
        frame = socket.recv(36)

        # The WAX9 device transmits SLIP-encoded data: packets begin and end
        # with 0xC0. So if the frame doesn't end in 0xC0, we haven't received
        # the entire packet yet.
        if not frame[-1] == '\xc0':
            frame = prev_frame + frame
            prev_frame = frame
        if frame[-1] == '\xc0' and len(frame) > 1:
            # When we reach the end of a packet, reset prev_frame and cycle to
            # the next device
            dev_id = dev_names.next()
            socket = devices[dev_id]
            prev_frame = ''
            # Make sure the packet we're about to write begins with 0xC0. Else we
            # lost part of it somewhere. Only decode complete packets.
            # FIXME: Warn or something when we lose a packet
            if frame[0] == '\xc0':
                line = [0] * 11
                if frame[2] == '\x01' and len(frame) == 28:
                    # Convert data from hex representation (see p. 7, 'WAX9 application
                    # developer's guide')
                    #print('01 | {} | {}'.format(len(frame), frame.encode('hex')))
                    line = list(struct.unpack('<BBBhIhhhhhhhhhB', frame))
                    line = line[3:14]
                elif frame[2] == '\x02' and len(frame) == 36:
                    # Convert data from hex representation (see p. 7, 'WAX9 application
                    # developer's guide')
                    #print('02 | {} | {}'.format(len(frame), frame.encode('hex')))
                    line = list(struct.unpack('<BBBhIhhhhhhhhhhhIB', frame))
                    line = line[3:14]
                else:
                    print('ER | {} | {}'.format(len(frame), frame.encode('hex')))
                line.append(time.time())
                line.append(dev_id)
                q.put(line)
        
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


def processData(path, dev_names):
    """
    Convert raw data to numpy array and save

    Args:
    -----
    [str] path: Path (full or relative) to raw data file
    [list(str)] dev_names: List of IMU ID strings; define m as the number of
      items in this list

    Returns:
    --------
    [np array] imu_data: Size n-by-m*d, where n is the number of IMU
      samples, m is the number of IMU devices, and d is the length of one IMU
      sample. Samples
    [np vector] rgb_data: Contains the global timestamp for every video
      frame recorded
    [int] sample_len: Length of one IMU sample (d)
    """
    
    num_devices = len(dev_names)
    name2idx = {dev_names[i]: i for i in range(num_devices)}

    imu_data = []
    rgb_data = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            sample_idx = int(row[0])
            timestamp = float(row[-2])

            # Rows w/ length 3 hold RGB frame data
            if len(row) == 3:
                for i in range(sample_idx - (len(rgb_data) - 1)):
                    rgb_data.append(0.0)
                rgb_data[sample_idx] = timestamp

            # Rows w/ length 13 hold IMU data
            if len(row) == 13:
                sample = [timestamp] + [int(x) for x in row[1:-2]]
                sample_len = len(sample)
                
                # Anything indexed below 1 is bad data
                # Any sample indexed more than 200 samples beyond the latest is
                # almost definitely bad data
                if sample_idx < 1 or sample_idx - len(imu_data) > 200:
                    continue
                
                # Append rows of zeros to the data matrix until the last row
                # is the sample index. Then write the current sample to its
                # place in the sample index.
                dev_idx = name2idx[row[-1]]
                for i in range(sample_idx - (len(imu_data) - 1)):
                    imu_data.append([[0.0] * sample_len] * num_devices)
                imu_data[sample_idx][dev_idx] = sample
    
    # Flatten nested lists in each row of IMU data
    imu_data = [[datum for sample in row for datum in sample] for row in imu_data]
    return (np.array(imu_data), np.array(rgb_data), sample_len)


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
    #addresses = ('00:17:E9:D7:09:5D',)
                 #'00:17:E9:D7:09:0F',
                 #'00:17:E9:D7:09:49')

    # Connect to devices and print settings
    devices = {}
    dev_settings = {}
    for address in addresses:
        print('Connecting at {}...'.format(address))
        # TODO: check if name is empty string and retry if so
        socket, name = connect(address)
        print('Connected, device ID {}'.format(name))
        settings = getSettings(socket)
        print(settings)
        devices[name] = socket
        dev_settings[name] = settings

    # Start receiving and writing data
    q = SimpleQueue()
    die = mp.Event()
    processes = []
    processes.append(mp.Process(target=streamImu, args=(devices, q, die)))
    processes.append(mp.Process(target=streamVideo, args=(rgb_trial_path, q, die)))
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
    for dev_name, socket in devices.items():
        print('Disconnecting from {}'.format(dev_name))
        socket.close()
    
    # Massage and save IMU, RGB frame data
    saveSettings(settings_file_path, dev_settings)
    imu_data, rgb_data, sample_len = processData(raw_file_path, devices.keys())
    fmtstr = len(devices) * (['%15f'] + (sample_len - 1) * ['%i'])
    np.savetxt(imu_file_path, imu_data, delimiter=',', fmt=fmtstr)
    np.savetxt(rgb_timestamp_path, rgb_data, delimiter=',', fmt='%15f')
    
    # Convert rgb frames to avi video
    # NOTE: This depends on the avconv utility for now
    print('')
    frame_fmt = os.path.join(rgb_trial_path, 'rgb_%6d.png')
    video_path = os.path.join(rgb_path, init_time + '.avi')
    make_video = ['avconv', '-f', 'image2', '-i', frame_fmt, '-r', '30', video_path]
    subprocess.call(make_video)
    print('')
    
    percent_dropped = printPercentDropped(imu_data, devices.keys(), sample_len)
    
    # Show IMU data (for validation)
    ids = [x[-4:] for x in devices.keys()]  # Grab hex ID from WAX9 ID
    plotImuData(int(init_time), ids)
    