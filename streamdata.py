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
from itertools import cycle
from libwax9 import *


def mpStreamVideo(q, die):
    """
    Stream data from camera until die is set

    Args:
    -----
      q: Multiprocessing queue, for collecting data
      die: Multiprocessing event, kill signal
    """

    dirname = str(int(time.time()))
    os.mkdir(dirname)

    openni2.initialize()
    dev = openni2.Device.open_any()

    # Start streaming color video
    color_stream = dev.create_color_stream()
    color_stream.start()

    """
    # Start streaming depth video
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    """

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
        filename = str(i) + "_depth" + ".png"
        cv2.imwrite(os.path.join(dirname, filename), depth_array)
        q.put((filename, frametime))
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
        filename = "rgb_" + str(i)  + ".png"
        cv2.imwrite(os.path.join(dirname, filename), color_array)
        q.put((i, frametime, "IMG_RGB"))

        i += 1

    # Stop streaming
    #depth_stream.stop()
    color_stream.stop()
    openni2.unload()


def mpStreamImu(devices, q, die):
    """
    Stream data from WAX9 devices until die is set

    Args:
    -----
      devices: Dict w/ device names as keys and their sockets as values
      q: Multiprocessing queue, for collecting data
      die: Multiprocessing event, kill signal
    """

    # This allows us to cycle through device names -> sockets in the main loop
    dev_names = cycle(devices.keys())

    # Tell the device to start streaming
    for socket in devices.values():
        socket.sendall("stream\r\n")

    # Read data sequentially from sensors until told to terminate
    stream = []
    prev_frame = ''
    dev_id = dev_names.next()
    socket = devices[dev_id]
    while True:
        # Wait 1 second before timing out
        # FIXME: It might not be necessary to check if the socket's ready to
        #   be read in this application
        #read_ready, wr, er = select.select([socket], [], [], 1)
        #if len(read_ready) == 0:
        #    continue
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
                # Convert data from hex representation (see p. 7, 'WAX9 application
                # developer's guide')
                line = list(struct.unpack("<hIhhhhhhhhh", frame[3:27]))
            else:
                line = 11 * [0]
            line[0] = line[0]
            line.append(time.time())
            line.append(dev_id)
            q.put(line)
        
        # Terminate when main tells us to
        if die.is_set():
            print("Dying...")
            # Tell the device to stop streaming
            for socket in devices.values():
                socket.sendall("\r\n")
            break


def mpWrite(fname, q, die):
    """
    Write data on queue to file until a kill signal is received. Once the kill
    signal is received, write what's left on the queue, then terminate

    Args:
    -----
      fname (str): output file name
      q: Multiprocessing queue, for collecting data
      die: Multiprocessing event, kill signal
    """

    with open(fname, 'wb') as csvfile:
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


def processData(filename, dev_names):
    """
    [DESCRIPTION]

    Args:
    -----
      filename (str):
      dev_names (list(str)):

    Returns:
    --------
      imu_data: numpy array
      rgb_data: numpy vector
      sample_len (int):
    """
    
    num_devices = len(dev_names)
    name2idx = {dev_names[i]: i for i in range(num_devices)}

    imu_data = []
    rgb_data = []
    with open(filename, 'r') as csvfile:
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

                dev_idx = name2idx[row[-1]]
                for i in range(sample_idx - (len(imu_data) - 1)):
                    imu_data.append([[0.0] * sample_len] * num_devices)
                imu_data[sample_idx][dev_idx] = sample
    
    # Flatten nested lists in each row of IMU data and cast to tuple
    imu_data = [[datum for sample in row for datum in sample] for row in imu_data]
    return (np.array(imu_data), np.array(rgb_data), sample_len)


def printPercentDropped(imu_data, dev_names, sample_len):
    """
    [DESCRIPTION]

    Args:
    -----
      imu_data:
      dev_names:
      sample_len:

    Returns:
    --------
      percent_dropped:
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
        fmtstr = "{}: {:0.1f}% of {} samples dropped"
        print(fmtstr.format(name, percent_dropped, num_samples))

        total_dropped += num_dropped
        total_samples += num_samples

    # Percent of samples dropped over all devices
    percent_dropped = total_dropped / total_samples * 100
    fmtstr = "TOTAL: {:0.1f}% of {} samples dropped"
    print(fmtstr.format(percent_dropped, int(total_samples)))

    return percent_dropped


if __name__ == "__main__":
    
    # raw_input() in python 2.x is input() in python 3.x
    # Below is for compatibility with python 2
    if sys.version_info[0] == 2:
        input = raw_input

    addresses = ("00:17:E9:D7:08:F1",
                 "00:17:E9:D7:09:5D",
                 "00:17:E9:D7:09:0F",
                 "00:17:E9:D7:09:49")

    init_time = int(time.time())

    # Connect to devices and print settings
    devices = {}
    for address in addresses:
        print("Connecting at {}...".format(address))
        # FIXME: check if name is empty string and redo if so
        socket, name = connect(address)
        print("Connected, device ID {}".format(name))
        print(getSettings(socket))
        devices[name] = socket

    # Stream data
    q = SimpleQueue()
    die = mp.Event()
    processes = []
    processes.append(mp.Process(target=mpStreamImu, args=(devices, q, die)))
    processes.append(mp.Process(target=mpStreamVideo, args=(q, die)))
    fname = "rawdata_{}.csv".format(init_time)
    processes.append(mp.Process(target=mpWrite, args=(fname, q, die)))

    for p in processes:
        p.start()

    # Wait for kill signal from user
    while True:
        user_in = input("Streaming... (press return to stop)")
        if not user_in:
            print("Killing processes...")
            die.set()
            break
    
    for p in processes:
        p.join()

    # Disconnect from devices
    for dev_name, socket in devices.items():
        print("Disconnecting from {}".format(dev_name))
        socket.close()

    # Massage and save data
    imu_data, rgb_data, sample_len = processData(fname, devices.keys())
    fname = 'imudata_{}.csv'.format(init_time)
    fmtstr = len(devices) * (['%15f'] + (sample_len - 1) * ['%i'])
    np.savetxt(fname, imu_data, delimiter=',', fmt=fmtstr)
    fname = 'rgbdata_{}.csv'.format(init_time)
    np.savetxt(fname, rgb_data, delimiter=',', fmt='%15f')

    percent_dropped = printPercentDropped(imu_data, devices.keys(), sample_len)
