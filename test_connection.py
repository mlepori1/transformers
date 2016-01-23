"""
test_connection.py
  Connect to four of the WAX9 IMUs, test of the functions in libwax9

AUTHOR
  Jonathan D. Jones
"""

import multiprocessing as mp
from libwax9 import *


addresses = ("00:17:E9:D7:08:F1",
             "00:17:E9:D7:09:5D",
             "00:17:E9:D7:09:0F",
             "00:17:E9:D7:09:49")
channel = 1

# Connect to devices and print settings
sockets = []
for address in addresses:
    socket = connect(address, channel)
    print(getSettings(socket))
    sockets.append(socket)

# Stream data using one process for each WAX9 device.
# (See 'An introduction to parallel programming using python's multiprocessing
# module', sebastian raschka, sebastianraschka.com)
q = mp.Queue()
processes = [mp.Process(target=stream, args=(s, 3, q)) for s in sockets]
for p in processes:
    p.start()
for p in processes:
    p.join()

# print streamed data
for p in processes:
    data = q.get()
    for sample in data:
        print(', '.join([str(x) for x in sample]))

# Disconnect from devices
print('Disconnecting')
for socket in sockets:
    socket.close()

