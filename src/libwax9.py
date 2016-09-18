"""
libwax9.py
  Library for interfacing with Axivity WAX9 IMU devices over bluetooth

AUTHOR
  Jonathan D. Jones
"""

import bluetooth
import select
import struct

import time
import csv

import numpy as np


def recvAll(socket, timeout):
    """
    Receive data from a WAX9 device.

    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device
    timeout: float 
      Amount of time to wait for a response, in seconds

    Returns:
    --------
    response: str
      Data sent by device
    """

    response = ''
    while True:
        # Wait 1 second before timing out
        read_ready, wr, er = select.select([socket], [], [], timeout)
        if len(read_ready) == 0:
            break
        data = socket.recv(1024)
        response += data

    return response


def connect(target_address):
    """
    Connect to a WAX9 IMU sensor over bluetooth.

    Args:
    -----
    target_addresss: str
      The sensor's MAC address, format 'XX:XX:XX:XX:XX'
    
    Returns:
    --------
    socket: bluetooth socket
      Connection to WAX9 device
    name: str
      Sensor ID
    """

    # I think the WAX9 sensors can only connect on channel 1 (whatever that
    # means)
    channel = 1

    name = bluetooth.lookup_name(target_address)
    socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    
    try:
        socket.connect((target_address, channel))
    except bluetooth.BluetoothError as e:
        print(e)
        socket = None
        name = None

    return (socket, name)


def getSettings(socket):
    """
    Read sensor settings from WAX9 device.

    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device

    Returns:
    --------
    settings: str
      WAX9 sensor settings
    """

    socket.sendall("settings\r\n")
    settings = recvAll(socket, 1)

    return settings


def parseSettings(settings_str):
    """
    Parse a WAX9 settings string
    
    Args:
    -----
    settings_str: str
      WAX9 sensor settings (e.g. output of getSettings)
    
    Returns:
    --------
    parsed_settings: numpy array
      length-1 structured array with these fields
      name: str
      mac: str
      accel on: int
      accel rate: int
      accel range: int
      gyro on: int
      gyro rate: int
      gyro range: int
      mag on: int
      mag rate: int
      ratex: int
      data mode: int
      sleep mode: int
      threshold: int
    """
    
    int_labels = ('ratex', 'data mode', 'sleep mode', 'threshold')
    sensors = ('accel', 'gyro', 'mag')
    
    setting_types = [('name', 'U10'), ('mac', 'U17'), ('accel on', 'i4'),
                     ('accel rate', 'i4'), ('accel range', 'i4'),
                     ('gyro on', 'i4'), ('gyro rate', 'i4'),
                     ('gyro range', 'i4'), ('mag on', 'i4'),
                     ('mag rate', 'i4'), ('ratex', 'i4'), ('data mode', 'i4'),
                     ('sleep mode', 'i4'), ('inactive', 'i4'),
                     ('threshold', 'i4')]
    parsed_settings = np.zeros(1, dtype=setting_types)
    
    lines = settings_str.split('\n')
    lines = [s1.strip().split(': ') for s1 in lines[2:9]] \
          + [s2.strip().split(':') for s2 in lines[9:]]
    
    for line in lines:
        var_type = line[0].lower()
        
        if var_type == 'name':
            data = line[1].split(', ')[0]
            parsed_settings[var_type] = data
        elif var_type == 'mac':
            data = line[1]
            parsed_settings[var_type] = data
        elif var_type in int_labels:
            data = int(line[1])
            parsed_settings[var_type] = data
        elif var_type in sensors:
            sensor_settings = line[1].split(', ')
            parsed_settings[var_type + ' on'] = sensor_settings[0]
            parsed_settings[var_type + ' rate'] = sensor_settings[1]
            if not var_type == 'mag':
                parsed_settings[var_type + ' range'] = sensor_settings[2]
        elif var_type == 'inactive':
            setting_str = line[1].split(', ')[0]
            assert(setting_str[-3:] == 'sec')
            parsed_settings[var_type] = int(setting_str[:-3])
    
    return parsed_settings


def sample(socket):
    """
    Read one data sample from WAX9 device.

    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device

    Returns:
    --------
    sample: str
      Strings of IMU samples
    """

    socket.sendall("sample\r\n")
    sample = recvAll(socket, 1)

    return sample
    

def stream(connected_devices, path, die, q):
    """
    Stream data from WAX9 devices until die is set. This function is intended
    to be used in a multithreading / multiprocessing setting.

    Args:
    -----
    connected_devices: dict of str -> bluetooth socket
      Dictionary mapping each WAX9 device's socket connections to its ID string 
    path: str
      Path to raw output file
    die: multiprocessing event
      The streaming loop monitors this event and quits when it is set
    q: multiprocessing queue
      Queue used to communicate with the main process.
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
    num_consec_timeouts = {dev_name: 0 for dev_name in connected_devices.keys()}

    # Tell the imu devices to start streaming
    for imu_socket in connected_devices.values():
        imu_socket.sendall('stream\r\n')
    
    # Terminate when main tells us to -- but only after we finish reading
    # the current set of packets
    while not die.is_set():
        
        # Use this list to store samples that are sent over the multiprocessing
        # queue
        samples = []
        
        # Read data sequentially from sensors until told to terminate
        for dev_id, imu_socket in connected_devices.items():
            
            # Read bytes from the current sensor until we have a complete
            # packet or hang on a timeout
            prev_escaped = False
            packet_ready = False
            timeout = False
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
                            # Anything else means we received an unexpected
                            # escaped byte
                            msg = 'ERR | {} | unexpected byte'
                            print(msg.format(byte.encode('hex')))
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
                    
                    frame = ''
                    packet_ready = True
                    prev_escaped = False
                    timeout = True
                    
                    num_consec_timeouts[dev_id] += 1
                    if num_consec_timeouts[dev_id] > MAX_NUM_TIMEOUTS:
                        die.set()
                        fmtstr = '\nDevice {} is unresponsive. Data collection' \
                                 ' halted. Cycle devices and restart software.'
                        print(fmtstr.format(dev_id))
                
            # Once we read a complete packet, unpack the bytes and write the
            # sample to file
            cur_time = time.time()
            
            if timeout:
                timeouts = num_consec_timeouts[dev_id]
                error = 1
                fmtstr = 'ERR | timeout  ({} consecutive) | {} | {}'
                print(fmtstr.format(timeouts, dev_id, packet.encode('hex')))
                data = [0] * 14
                data[1] = error
                data[-1] = dev_id
                timeout = False
            elif packet[2] == '\x01' and len(packet) == 28:  # Standard packet
                error = 0
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                # Zero-pad this packet at the end so it's the same length
                # and format as the long packet
                fmtstr = '<' + 3 * 'B' + 'H' + 'I' + 9 * 'h' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data = [cur_time, error] + unpacked[3:14] + 3 * [0] + [dev_id]
            elif packet[2] == '\x02' and len(packet) == 36:  # Long packet
                error = 0
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                fmtstr = '<' + 3 * 'B' + 'H' + 'I' + 9 * 'h' + 'H' + 'h' + 'I' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data = [cur_time, error] + unpacked[3:17] + [dev_id]
            else:
                error = 1
                # Record an error
                fmtstr = 'ERR | Bad packet | {} | {}'
                print(fmtstr.format(dev_id, packet.encode('hex')))
                data = [0] * 17
                data[1] = error
                data[-1] = dev_id
            
            samples.append(data)
            
            # Write the packet
            imu_writer.writerow(data)
        
        # Put a new sample set on the queue if the previous one has been
        # consumed
        if q.empty():
            q.put(samples)
        
        
    # Tell the devices to stop streaming and close the file we were writing
    for imu_socket in connected_devices.values():
        imu_socket.sendall('\r\n')
    f_imu.close()


def setDataMode(socket, datamode):
    """
    Set WAX9 device's data streaming mode.

    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device
    datamode: int
      Format to be sent when streaming data (this only affects data received
      from stream(), not sample())
      0 ---- Ascii mode with batt, temp, pressure, inactivity transmitted as
             they are sampled (~1Hz). 
      1 ---- Binary mode with ~1Hz batt, temp, pressure, inactivity update
             transmitted as they are sampled.
      128 -- Ascii mode with batt, temp, pressure, inactivity transmitted every
             packet.
      129 -- Binary mode with continuous batt, temp, pressure, inactivity
             update transmitted every packet.

    Returns:
    --------
    settings: str
      Device's settings after data mode has been changed.
    """

    socket.sendall("datamode {}\r\n".format(datamode))
    settings = recvAll(socket, 1)

    return settings


def setRate(socket, rate):
    """
    Set WAX9 device's output rate.

    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device
    rate: int
      New output rate in samples per second (Hz) (default is 50Hz)

    Returns:
    --------
    settings: str
      Device's settings after rate has been changed
    """

    socket.sendall("rate x 0 0 {}\r\n".format(rate))
    settings = recvAll(socket, 1)

    return settings


def setThreshold(socket, threshold):
    """
    Set the number of samples to log before transmitting a packet.
    
    Args:
    -----
    socket: bluetooth socket
      Connection to WAX9 device
    threshold: int
      Number of samples to log before transmitting a packet
    
    Returns:
    --------
    settings: str
      Device's settings after threshold has been changed
    """
    
    socket.sendall("threshold {}\r\n".format(threshold))
    settings = recvAll(socket, 1)
    
    return settings


def newStream(devices, names, filename):

    dev2name = {device: name for device, name in zip(devices, names)}
    prev_parsed = {name: '' for name in names}
    
    import datetime
    
    #f = open(filename, 'wb')
    
    for device in devices:
        device.sendall('stream\r\n')
    
    """
    ready, _, _ = select.select(devices, [], [])
    while ready:
        for device in ready:
            byte_str, addr = device.recvfrom(4096)
            f.write(byte_str)
            cur_time = datetime.datetime.now()
            prev_time = cur_time
            print(len(byte_str))
            print(byte_str.encode('hex'))
        ready, _, _ = select.select([device], [], [], 0)
    """
    prev_time = datetime.datetime.now()
    
    ready, _, _ = select.select(devices, [], [])
    while ready:
        for device in ready:
            name = dev2name[device]
            
            byte_str, addr = device.recvfrom(4096)
            data, incomplete = parseBytes(byte_str, prev_parsed[name])
            prev_parsed[name] = incomplete
            #f.write(byte_str)
            
            cur_time = datetime.datetime.now()
            print('{}  |  {}  |  {}'.format(name, cur_time - prev_time, len(byte_str)))
            for datum in data: print(datum) #.encode('hex'))
            print('')
            prev_time = cur_time
        ready, _, _ = select.select(devices, [], []) #, 0)
    
    device.sendall('\r\n')    
    
    #f.close()


def parseBytes(byte_str, prev_parsed):
        
    # Special SLIP bytes
    SLIP_END = '\xc0'
    SLIP_ESC = '\xdb'
    SLIP_ESC_END = '\xdc'
    SLIP_ESC_ESC = '\xdd'
    
    data = []
    
    # Read bytes from the current sensor until we have a complete
    # packet or hang on a timeout
    prev_escaped = False
    #packet_ready = False
    #timeout = False
    packet = prev_parsed
    for byte in byte_str:                
        
        if prev_escaped:
            prev_escaped = False
            if byte == SLIP_ESC_END:
                packet += SLIP_END
            elif byte == SLIP_ESC_ESC:
                packet += SLIP_ESC
            else:
                # Anything else means we received an unexpected
                # escaped byte
                msg = 'ERR | {} | unexpected byte'
                print(msg.format(byte.encode('hex')))
        elif byte == SLIP_END and len(packet) > 0:
            packet += byte
            if packet[2] == '\x01' and len(packet) == 28:  # Standard packet
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                # Zero-pad this packet at the end so it's the same length
                # and format as the long packet
                fmtstr = '<' + 3 * 'B' + 'H' + 'I' + 9 * 'h' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data.append(unpacked)
            elif packet[2] == '\x02' and len(packet) == 36:  # Long packet
                # Convert data from hex representation (see p. 7, 'WAX9
                # application developer's guide')
                fmtstr = '<' + 3 * 'B' + 'H' + 'I' + 9 * 'h' + 'H' + 'h' + 'I' + 'B'
                unpacked = list(struct.unpack(fmtstr, packet))
                data.append(unpacked)
            else:
                # Record an error
                fmtstr = 'ERR | Bad packet | {}'
                print(fmtstr.format(packet.encode('hex')))
            packet = ''
        elif byte == SLIP_ESC:
            prev_escaped = True
        else:
            packet += byte
        
    return data, packet


def decodeStream(filename):
    
    with open(filename, 'rb') as f:
        for line in f:
            print(line.encode('hex'))


if __name__ == '__main__':
    
    filename = 'test'
    target_addresses = ('00:17:E9:D7:09:49', '00:17:E9:D7:09:0F',
                        '00:17:E9:D7:09:5D', '00:17:E9:D7:08:F1')
    rate = 150
    threshold = 512
    
    devices = {}
    for address in target_addresses:
        socket, name = connect(address)
        devices[name] = socket
        print(name)
    
    for socket in devices.values():
        settings = getSettings(socket)
        print(settings)
    
    for socket in devices.values():
        settings = setRate(socket, rate)
        print(settings)
    
    for socket in devices.values():
        settings = setThreshold(socket, threshold)
        print(settings)
    
    newStream(devices.values(), devices.keys(), filename)
    
    decodeStream(filename)