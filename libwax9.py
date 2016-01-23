"""
libwax9.py
  Library for interfacing with Axivity WAX9 IMU devices over bluetooth

AUTHOR
  Jonathan D. Jones
"""

import bluetooth
import select
import time
import sys
import struct


def recvAll(socket, timeout):
    """
    Receive data from a WAX9 device

    Args:
    -----
      socket: WAX9 bluetooth socket
      timeout (float): Amount of time to wait for a response, in seconds

    Returns:
    --------
      response (str): Data sent by device
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


def connect(target_address, channel):
    """
    Connect to a WAX9 IMU sensor over bluetooth

    Args:
    -----
      target_addresss (str): The sensor's MAC address, format 'XX:XX:XX:XX:XX'
    
    Returns:
    --------
      socket: Client connection to the sensor (bluetooth socket)
    """

    name = bluetooth.lookup_name(target_address)
    print("Connecting to {} at {} on channel {}...".format(name, target_address, channel))
    socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    socket.connect((target_address, channel))
    print("Connected")

    return socket


def getSettings(socket):
    """
    Read sensor settings from WAX9 device

    Args:
    -----
      socket: WAX9 bluetooth socket

    Returns:
    --------
      settings (str): WAX9 sensor settings
    """

    print("Reading settings...")
    socket.sendall("settings\r\n")
    settings = recvAll(socket, 1)

    return settings


def sample(socket):
    """
    Read one data sample from WAX9 device

    Args:
    -----
      socket: WAX9 bluetooth socket

    Returns:
    --------
      sample (str): Strings of IMU samples
    """

    print("Reading a sample...")
    socket.sendall("sample\r\n")
    sample = recvAll(socket, 1)

    return sample


def stream(socket, duration, q):
    """
    Stream data from WAX9 device for `duration` seconds

    Args:
    -----
      socket: WAX9 bluetooth socket
      duration (float): Amount of time (in seconds) to stream

    Returns:
    --------
      sample (str): Strings of IMU samples
    """

    print("Streaming...")
    # Tell the device to start streaming
    socket.sendall("stream\r\n")

    # Read data for two seconds
    stream = []
    prev_frame = ''
    start_time = time.time()
    while time.time() - start_time < duration:
        # Wait 1 second before timing out
        read_ready, write_ready, exept_ready = select.select([socket], [], [], 1)
        if len(read_ready) == 0:
            break
        frame = socket.recv(36)
        if not frame[-1] == '\xc0':
            frame = prev_frame + frame
            prev_frame = frame
        if frame[-1] == '\xc0' and len(frame) > 1:
            prev_frame = ''
            #print('{}'.format(repr(frame)))
            #print('')
            if frame[0] == '\xc0':
                # Convert data from hex representation (see p. 7, 'WAX9 application
                # developer's guide')
                data = list(struct.unpack("<hIhhhhhhhhh", frame[3:27]))
                data.append(time.time())
                stream.append(data)
            #    sys.stdout.write(str(time.time()) + '\t' + str(data))

    #print('')

    # Tell the device to stop streaming
    socket.sendall("\r\n")

    q.put(stream)


def setDataMode(socket, datamode):
    """
    Set WAX9 device's data streaming mode. This is mostly for debug use.

    Args:
    -----
      socket: WAX9 bluetooth socket
      datamode (int): Acceptable values are 0, 1, 128, 129

    Returns:
    --------
      settings (str): Device's settings
    """

    socket.sendall("datamode {}\r\n".format(datamode))
    settings = recvAll(socket, 1)

    return settings


def setRate(socket, rate):
    """
    Set WAX9 device's output rate. This is mostly for debug use.

    Args:
    -----
      socket: WAX9 bluetooth socket
      rate (int): new output rate (default is 50Hz)

    Returns:
    --------
      settings (str): Device's settings
    """

    socket.sendall("rate x 0 0 {}\r\n".format(rate))
    settings = recvAll(socket, 1)

    return settings

