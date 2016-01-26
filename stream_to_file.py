"""
stream_to_file.py
  Connect to four WAX9 IMUs, stream data, write to CSV

AUTHOR
  Jonathan D. Jones
"""

import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
import csv
import time
from libwax9 import *

def mpStream(socket, q, die):
    """
    Stream data from WAX9 device until die is set

    Args:
    -----
      socket: WAX9 bluetooth socket
      q: Multiprocessing queue, for collecting data
      die: Multiprocessing event, kill signal
    """

    # Name the current process after the device we're streaming from
    dev_id = mp.current_process().name

    # Tell the device to start streaming
    socket.sendall("stream\r\n")

    # Read data until told to terminate
    stream = []
    prev_frame = ''
    while True:
        # Wait 1 second before timing out
        # FIXME: It might not be necessary to check if the socket's ready to
        #   be read in this application
        read_ready, wr, er = select.select([socket], [], [], 1)
        if len(read_ready) == 0:
            continue
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
                line.append(time.time())
                line.append(dev_id)
                q.put(line)
        
        # Terminate when main tells us to
        if die.is_set():
            print("Dying...")
            # Tell the device to stop streaming
            socket.sendall("\r\n")
            break


def mpWrite(fname, q, die):
    """
    Write data on queue to file until a kill signal is received. Once the kill
    signal is received, write what's left on the queue, then terminate

    Args:
    -----
      fname (str): output file name
      q: Multiprocessing queue
      die:
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


if __name__ == "__main__":
    
    # raw_input() in python 2.x is input() in python 3.x
    # Below is for compatibility with python 2
    if sys.version_info[0] == 2:
        input = raw_input

    fname = "imu-data_{}.csv".format(int(time.time()))

    addresses = ("00:17:E9:D7:08:F1",
                 "00:17:E9:D7:09:5D",
                 "00:17:E9:D7:09:0F",
                 "00:17:E9:D7:09:49")

    # Connect to devices and print settings
    devices = {}
    for address in addresses:
        print("Connecting at {}...".format(address))
        socket, name = connect(address)
        print("Connected, device ID {}".format(name))
        print(getSettings(socket))
        devices[name] = socket

    # Stream data using one process for each WAX9 device
    q = SimpleQueue()
    die = mp.Event()
    processes = []
    for dev_name, socket in devices.items():
        p = mp.Process(name=dev_name, target=mpStream, args=(socket, q, die))
        processes.append(p)
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

