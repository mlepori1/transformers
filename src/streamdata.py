"""
streamdata.py
  Connect to four WAX9 IMUs and a PrimeSense camera, stream data, write to CSV

AUTHOR
  Jonathan D. Jones
"""

import numpy as np
import multiprocessing as mp
import select
from multiprocessing.queues import SimpleQueue
from primesense import openni2
import cv2
import csv
import os
import sys
import time
import struct

from libwax9 import *
from libduplo import *
from duplocorpus import DuploCorpus


def streamVideo(dev_name, die, path):
    """
    Stream data from camera until die is set

    Args:
    -----
    [dict(str->cv stream)] dev_name:
    [mp event] die: Multiprocessing event, kill signal
    [str] path: Path (full or relative) to image output directory
    """
    
    f_rgb = open(os.path.join(path, 'frame_timestamps.csv'), 'w')
    rgb_writer = csv.writer(f_rgb)
    
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
        rgb_writer.writerow((frametime, frame_index, dev_name))

        frame_index += 1

    # Stop streaming
    print("Closing RGB camera")    
    stream.stop()
    openni2.unload()
    
    f_rgb.close()
    

def streamImu(devices, die, path):
    """
    Stream data from WAX9 devices until die is set

    Args:
    -----
    [dict(str->socket)] devices: Dict w/ device names as keys and their
      sockets as values
    [mp event] die: Multiprocessing event, kill signal
    [str] path: Path to raw IMU output file
    """
    
    SAMPLE_RATE = 15
    TIMEOUT = 1.5 * (1.0 / SAMPLE_RATE)
    
    # If we see more than this many timeouts, it means we've dropped samples
    # for three seconds consecutively. The IMU is probably stalled in this
    # case, so stop data collection and prompt user to cycle the device.
    MAX_NUM_TIMEOUTS = 3 * SAMPLE_RATE
    
    # Set up output files
    f_imu = open(path, 'w')
    imu_writer = csv.writer(f_imu)
    
    # Special SLIP bytes
    SLIP_END = '\xc0'
    SLIP_ESC = '\xdb'
    SLIP_ESC_END = '\xdc'
    SLIP_ESC_ESC = '\xdd'
    
    # This dictionary is used to track the number of consecutive dropped
    # samples for each IMU device
    num_consec_timeouts = {dev_name: 0 for dev_name in devices.keys()}

    # Tell the imu devices to start streaming
    for imu_socket in devices.values():
        imu_socket.sendall('stream\r\n')
    
    while True:
        
        # Read data sequentially from sensors until told to terminate
        for dev_id, imu_socket in devices.items():   
            
            # Read bytes from the current sensor until we have a complete
            # packet or hang on a timeout
            prev_escaped = False
            packet_ready = False
            frame = ''
            while not packet_ready:                
                # Time out if we spend longer than the sample period waiting to
                # read from the socket
                ready = select.select([imu_socket], [], [], TIMEOUT)
                if ready[0]:
                    # Reset the number of consecutive dropped samples once we
                    # successfully read one
                    num_consec_timeouts[dev_id] = 0
                    
                    # Read and process one byte from the IMU
                    byte = imu_socket.recv(1)
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
                # If select times out, we've waited longer than a sample
                # period. This sample must be bad, so throw it out by resetting
                # the data frame.
                else:
                    packet = frame
                    print('ERR | timeout | {}'.format(frame.encode('hex')))
                    frame = ''
                    packet_ready = True
                    prev_escaped = False
                    
                    num_consec_timeouts[dev_id] += 1
                    if num_consec_timeouts[dev_id] > MAX_NUM_TIMEOUTS:
                        die.set()
                        fmtstr = '\nDevice {} is unresponsive. Data collection' \
                                 'halted. Cycle devices and restart software.'
                        print(fmtstr.format(dev_id))
                        #sys.exit()
                
            # Once we read a complete packet, unpack the bytes and queue the data
            # to be written
            cur_time = time.time()
            
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
                data = [0] * 14
                data[1] = error
                data[-1] = dev_id
                
            # Queue the packet to be written
            imu_writer.writerow(data)
        
        # Terminate when main tells us to -- but only after we finish
        # reading the current set of packets
        if die.is_set():
            # Tell the devices to stop streaming
            for imu_socket in devices.values():
                imu_socket.sendall('\r\n')
            break
    
    f_imu.close()


if __name__ == "__main__":
    
    # 15 samples per second + a little room
    SAMPLE_FREQ = 15
    SAMPLE_PERIOD = 1.0 / SAMPLE_FREQ + 0.001
    
    # raw_input() in python 2.x is input() in python 3.x
    # Below is for compatibility with python 2
    if sys.version_info[0] == 2:
        input = raw_input
    
    # Prompt user for metadata
    MIN_AGE = 4
    MAX_AGE = 6
    child_age = int(input("What is the child's age? Enter a number between 4 and 6.\n>> "))
    while child_age < MIN_AGE or child_age > MAX_AGE:
        child_age = int(input("Please enter a number between 4 and 6.\n>> "))
    
    child_gender = input("What is the child's gender? Enter m, f, or o.\n>> ").lower()
    while not child_gender in ('m', 'f', 'o'):
        child_gender = input("Please enter m (male), f (female) or o "
                             "(other / do not wish to disclose).\n>> ").lower()
    
    num_blocks = int(input("How many blocks are in this task? Enter 4, 6, or 8.\n>> "))
    while not num_blocks in (4, 6, 8):
        num_blocks = int(input("Please enter 4, 6, or 8.\n>> "))
    if num_blocks == 4:
        block_colors = ('red', 'yellow')
    elif num_blocks == 6:
        block_colors = ('red', 'yellow', 'blue')
    else:
        block_colors = ('red', 'yellow', 'blue', 'green')
    
    imu2block = {}
    imu_devs = {}
    imu_settings = {}
    mac_prefix = ['00', '17', 'E9', 'D7']
    imu_ids = ('08F1', '095D', '090F', '0949')
    imus_in_use = []
    for block_color in block_colors:
        
        name = None
        while name is None:
        
            # Match IMUs to block colors
            while True:
                msg_str = "Place a sensor in the {} block. "\
                          "What is the sensor's ID?\n>> "
                imu_id = input(msg_str.format(block_color))
                while not imu_id in imu_ids:
                    msg_str = "That isn't a valid ID. Sensor IDs are: {}\n>> "
                    imu_id = input(msg_str.format(imu_ids))
                if not imu_id in imus_in_use:
                    imus_in_use.append(imu_id)
                    break
                print('That sensor is already in use!')
            
            imu2block[imu_id] = block_color
            
            # Compute the device's MAC address
            mac_bytes = mac_prefix + [imu_id[0:2], imu_id[2:4]]
            address = ':'.join(mac_bytes)
            
            # Connect to devices and print settings
            try:
                print('Connecting to device at {}...'.format(address))
                imu_socket, name = connect(address)
            except BluetoothError:
                name = None
            while name is None:     # Sometimes we get an empty connection
                user_in = input('Bad connection. Try again? Enter y or n.\n>>').lower()
                if user_in == 'n':
                    break
                try:
                    imu_socket, name = connect(address)
                except BluetoothError:
                    name = None
        print('Connected, device ID {}'.format(name))
        settings = getSettings(imu_socket)
        imu_devs[name] = imu_socket
        imu_settings[name] = settings
        
        """
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
            line = q.get()
            test_data.append(line[:-1])
        test_data = np.array(test_data)
        a_norm = np.sum(test_data[:,4:7] ** 2, 1)
        w_norm = np.sum(test_data[:,7:10] ** 2, 1)
        h_norm = np.sum(test_data[:,10:13] ** 2, 1)
        norm_data = np.vstack((test_data[:,0], a_norm, w_norm, h_norm)).T
        fig_text = ['Square L2 norm', '\| \cdot \|^2', '??']
        f = plot3dof(norm_data, np.array([0]), np.array([0, test_data.shape[0] - 1]), fig_text)
        plt.show()
        """
    # Record which IMUs weren't used
    for imu_id in imu_ids:
        if not imu_id in imus_in_use:
            imu2block[imu_id] = 'UNUSED'
    assert(len(imu2block) == len(imu_ids))
    
    corpus = DuploCorpus()
    
    # Define pathnames and filenames for the data from this trial
    trial_id = corpus.meta_data.shape[0]
    init_time = time.time()
    
    raw_file_path = os.path.join(corpus.paths['imu-raw'], '{}.csv'.format(trial_id))
    rgb_trial_path = os.path.join(corpus.paths['rgb'], str(trial_id))
    if not os.path.exists(rgb_trial_path):
        os.makedirs(rgb_trial_path)
    
    t_start = time.time()               
    
    # Start receiving and writing data
    img_dev_name = 'IMG-RGB'
    q = SimpleQueue()
    die = mp.Event()
    processes = (mp.Process(target=streamImu, args=(imu_devs, die, raw_file_path)),
                 mp.Process(target=streamVideo, args=(img_dev_name, die, rgb_trial_path)))
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
    for dev_name, imu_socket in imu_devs.items():
        print('Disconnecting from {}'.format(dev_name))
        imu_socket.close()
    
    # Estimate number of samples from duration of data collection
    t_end = time.time()
    print((t_end - t_start) * SAMPLE_FREQ)
    
    child_id = '_'.join((str(child_age), child_gender))
    corpus.postprocess(child_id, trial_id, imu_devs, imu_settings, img_dev_name, imu2block)        
        
    # Show IMU data (for validation)
    ids = [x[-4:] for x in imu_devs.keys()]  # Grab hex ID from WAX9 ID
    corpus.makeImuFigs(trial_id, ids)
    