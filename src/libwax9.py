"""
libwax9.py
  Library for interfacing with Axivity WAX9 IMU devices over bluetooth

AUTHOR
  Jonathan D. Jones
"""

import bluetooth
import select
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


def connect(target_address):
    """
    Connect to a WAX9 IMU sensor over bluetooth

    Args:
    -----
      target_addresss (str): The sensor's MAC address, format 'XX:XX:XX:XX:XX'
    
    Returns:
    --------
      socket: Client connection to the sensor (bluetooth socket)
      name (str): Sensor ID
    """

    # I think the WAX9 sensors can only connect on channel 1
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
    Read sensor settings from WAX9 device

    Args:
    -----
      socket: WAX9 bluetooth socket

    Returns:
    --------
      settings (str): WAX9 sensor settings
    """

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

    socket.sendall("sample\r\n")
    sample = recvAll(socket, 1)

    return sample


def stream(socket):
    """
    Stream data from WAX9 device until user inputs a newline

    Args:
    -----
      socket: WAX9 bluetooth socket

    Returns:
    --------
      sample (str): Strings of IMU samples
    """

    # Tell the device to start streaming
    socket.sendall("stream\r\n")

    # Read data for two seconds
    stream = []
    prev_frame = ''
    stop = False
    while True:
        # Wait 1 second before timing out
        read_ready, wr, er = select.select([socket], [], [], 1)
        if len(read_ready) == 0:
            break
        frame = socket.recv(36)
        # The WAX9 device transmits SLIP-encoded data: packets begin and end
        # with 0xC0. So if the frame doesn't end in 0xC0, we haven't received
        # the entire packet yet.
        if not frame[-1] == '\xc0':
            frame = prev_frame + frame
            prev_frame = frame
        # Make sure the packet we're about to write begins with 0xC0. Else we
        # lost part of it somewhere. Only decode complete packets.
        # FIXME: Warn or something when we lose a packet
        if frame[-1] == '\xc0' and len(frame) > 1:
            prev_frame = ''
            if frame[0] == '\xc0':
                # Convert data from hex representation (see p. 7, 'WAX9 application
                # developer's guide')
                line = list(struct.unpack("<hIhhhhhhhhh", frame[3:27]))
                stream.append(line)

        # Abort if user has entered a blank line
        if sys.stdin.read() == "\n":
            break

    # Tell the device to stop streaming
    socket.sendall("\r\n")

    return stream


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

