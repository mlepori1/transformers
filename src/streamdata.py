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
import struct

from libwax9 import *
from libduplo import *
from duplocorpus import DuploCorpus


def streamVideo(dev_name, q, die, path):
    """
    Stream data from camera until die is set

    Args:
    -----
    [dict(str->cv stream)] dev_name:
    [mp queue] q: Multiprocessing queue, for collecting data
    [mp event] die: Multiprocessing event, kill signal
    [str] path: Path (full or relative) to image output directory
    """
    
    # Open video streams
    print("Opening RGB camera...")
    openni2.initialize()
    dev = openni2.Device.open_any()
    stream = dev.create_color_stream()
    print("RGB camera opened")
    
    frame_index = 0
    
    print("Starting RGB stream...")
    stream.start()
    print("RGB stream started")
    
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
        
        # Read frame data, record frame timestamp
        frame = stream.read_frame()
        frametime = time.time()
        
        # Convert to image array
        data = frame.get_buffer_as_uint8()
        img_array = np.ndarray((frame.height, 3*frame.width),
                               dtype=np.uint8, buffer=data)
        img_array = np.dstack((img_array[:,2::3], img_array[:,1::3], img_array[:,0::3]))
        
        # Write to file
        filename = '{:06d}.png'.format(frame_index)
        cv2.imwrite(os.path.join(path, filename), img_array)
        q.put((frametime, frame_index, dev_name))

        frame_index += 1

    # Stop streaming
    print("Closing RGB camera")    
    stream.stop()
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
    
    # Special SLIP bytes
    SLIP_END = '\xc0'
    SLIP_ESC = '\xdb'
    SLIP_ESC_END = '\xdc'
    SLIP_ESC_ESC = '\xdd'

    # This allows us to cycle through device names -> sockets in the main loop
    dev_names = cycle(devices.keys())

    # Tell the device to start streaming
    for socket in devices.values():
        socket.sendall('stream\r\n')

    # Read data sequentially from sensors until told to terminate
    prev_escaped = False
    packet_ready = False
    frame = ''
    data = [0] * 14
    dev_id = dev_names.next()
    socket = devices[dev_id]
    while True:
        
        # Read and process one byte from the IMU
        byte = socket.recv(1)
        if prev_escaped:
            prev_escaped = False
            if byte == SLIP_ESC_END:
                frame += SLIP_END
            elif byte == SLIP_ESC_ESC:
                frame += SLIP_ESC
            else:
                # Anything else means we received an unexpected escaped byte
                print('ERR | {} | unexpected byte'.format(byte.encode('hex')))
        elif byte == SLIP_END and len(frame) > 0:
            frame += byte
            packet = frame
            frame = ''
            packet_ready = True
        elif byte == SLIP_ESC:
            prev_escaped = True
        else:
            frame += byte
        
        # Once we read a complete packet, unpack the bytes and queue the data
        # to be written
        if packet_ready:
            
            cur_time = time.time()
            packet_ready = False
            
            if len(packet) == 28:   # Standard packet
                error = 0
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                fmtstr = '<' + 3 * 'B' + 'h' + 'I' + 9 * 'h' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data = [cur_time, error] + unpacked[3:14] + [dev_id]
            elif len(packet) == 36: # Long packet
                error = 0
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                fmtstr = '<' + 3 * 'B' + 'h' + 'I' + 11 * 'h' + 'I' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data = [cur_time, error] + unpacked[3:14] + [dev_id]
            else:
                error = 1
                # Record an error
                print('ERR | {} | {} | {}'.format(dev_id, len(packet), packet.encode('hex')))
                data = [0] * len(data)
                data[1] = error
            
            # Queue the packet to be written
            q.put(data)
            
            # Cycle to the next device
            dev_id = dev_names.next()
            socket = devices[dev_id]
        
        # Terminate when main tells us to
        if die.is_set():
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


def testImu(dev_name, socket):
    """
    Verify IMU is functioning properly by instructing the user to lift it and
    put it back down.
    
    Args:
    -----
    [str] dev_name: Name of IMU device to be tested
    [socket] socket: Socket object representing bluetooth connection
    
    Returns:
    --------
    [bool] valid: Is the device working as expected? Boolean T/F
    """
    
    print('Pick up sensor {} and put it back down.')


if __name__ == "__main__":
    
    # raw_input() in python 2.x is input() in python 3.x
    # Below is for compatibility with python 2
    if sys.version_info[0] == 2:
        input = raw_input
    
    # Prompt user for metadata
    MIN_AGE = 4
    MAX_AGE = 6
    child_age = int(input("What is the child's age?\n>> "))
    while child_age < MIN_AGE or child_age > MAX_AGE:
        child_age = int(input("Please enter a number between 4 and 6.\n>> "))
    
    child_gender = input("What is the child's gender?\n>> ")
    while not child_gender in ('M', 'F', 'O'):
        child_age = input("Please enter M (male), F (female) or O "
                          "(other / do not wish to disclose).\n>> ")
    
    num_blocks = int(input("How many blocks are in this task?\n>> "))
    while not num_blocks in (4, 6, 8):
        num_blocks = int(input("Please enter 4, 6, or 8.\n>> "))
    if num_blocks == 4:
        block_colors = ('red', 'yellow')
    else:
        block_colors = ('red', 'yellow', 'blue', 'green')
    
    imu_devs = {}
    imu_settings = {}
    mac_prefix = ['00', '17', 'E9', 'D7']
    imu_ids = ('08F1', '095D', '090F', '0949')
    imus_in_use = []
    for block_color in block_colors:
        
        # Match IMUs to block colors
        while True:
            msg_str = "Place a sensor in the {} block. What is the sensor's ID?\n>> "
            imu_id = input(msg_str.format(block_color))
            while not imu_id in imu_ids:
                msg_str = "That isn't a valid ID. Sensor IDs are: {}\n>> "
                imu_id = input(msg_str.format(imu_ids))
            if not imu_id in imus_in_use:
                imus_in_use.append(imu_id)
                break
            print('That sensor is already in use!')
        
        # Compute the device's MAC address
        mac_bytes = mac_prefix + [imu_id[0:2], imu_id[2:4]]
        address = ':'.join(mac_bytes)
        
        # Connect to devices and print settings
        print('Connecting to device at {}...'.format(address))
        socket, name = connect(address)
        while name is None:     # Sometimes we get an empty connection
            print('Bad connection. Trying again...')
            socket, name = connect(address)
        print('Connected, device ID {}'.format(name))
        settings = getSettings(socket)
        print(settings)
        imu_devs[name] = socket
        imu_settings[name] = settings
        
        # Test device connection
        q = SimpleQueue()
        die = mp.Event()
        imu_dev = {name: socket}
        p = mp.Process(target=streamImu, args=(imu_dev, q, die))
        p.start()
        msg_str = 'Pick up the {} block and put it back down. '
        input(msg_str.format(block_color) + 'Then, press enter.\n>> ')
        die.set()
        p.join()
        
        # Print test stats
        test_data = []
        while not q.empty():
            test_data.append(q.get())
        test_data = np.array(test_data)
        test_avg = test_data.mean(axis=0)
        test_std = test_data.std(axis=0)
        print('Mean of IMU readings: {}'.format(test_avg))
        print('Standard deviation of IMU readings: {}'.format(test_std))
    
    corpus = DuploCorpus()
    
    # Define pathnames and filenames for the data from this trial
    trial_id = corpus.meta_data.shape[0]
    init_time = time.time()
    
    raw_file_path = os.path.join(corpus.paths['raw'], '{}.csv'.format(trial_id))
    rgb_trial_path = os.path.join(corpus.paths['rgb'], str(trial_id))
    if not os.path.exists(rgb_trial_path):
        os.makedirs(rgb_trial_path)                
    
    # Start receiving and writing data
    img_dev_name = 'IMG-RGB'
    q = SimpleQueue()
    die = mp.Event()
    processes = (mp.Process(target=streamImu, args=(imu_devs, q, die)),
                 mp.Process(target=streamVideo, args=(img_dev_name, q, die, rgb_trial_path)),
                 mp.Process(target=write, args=(raw_file_path, q, die)))
    for p in processes:
        p.start()

    # Wait for kill signal from user
    while True:
        user_in = input('Streaming... (press return to stop)\n')
        if not user_in:
            print('Killing processes')
            die.set()
            break
    
    for p in processes:
        p.join()

    # Disconnect from devices
    for dev_name, socket in imu_devs.items():
        print('Disconnecting from {}'.format(dev_name))
        socket.close()
    
    child_id = '_'.join((child_age, child_gender))
    corpus.postprocess(child_id, trial_id, imu_devs, imu_settings, img_dev_name)        
        
    # Show IMU data (for validation)
    ids = [x[-4:] for x in imu_devs.keys()]  # Grab hex ID from WAX9 ID
    corpus.makeImuFigs(trial_id, ids)
    