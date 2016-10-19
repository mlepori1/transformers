
from pymetawear.client import *
import time

def accelHandler(data, client_index):
    #epoch = data[0]
    #xyz = data[1]
    source_str = "CLIENT {} | A | ".format(client_index)
    #data_str = "{0} | X: {1}, Y: {2}, Z: {3}".format(epoch, *xyz)
    data_str = str(data)
    print(source_str + data_str)

def gyroHandler(data, client_index):
    #epoch = data[0]
    #xyz = data[1]
    source_str = "CLIENT {} | G | ".format(client_index)
    #data_str = "{0} | X: {1}, Y: {2}, Z: {3}".format(epoch, *xyz)
    data_str = str(data)
    print(source_str + data_str)

addresses = ('D6:B3:DA:FD:2E:DE',
             'FC:63:6C:B3:4C:F6',
             'F7:A1:FC:73:DD:23',
             'C7:7D:36:B1:5E:7D')
addresses = addresses[0:1]

clients = []
for a in addresses:
    print('Connecting to {}...'.format(a))
    clients.append(MetaWearClient(a, backend='pybluez'))

rate = 25
for c in clients:
    c.accelerometer.set_settings(data_rate=rate, data_range=8)
    c.gyroscope.set_settings(data_rate=rate, data_range=1000)
    #c.accelerometer.high_frequency_stream = True
    #c.gyroscope.high_frequency_stream = True

print('Subscribing...')
for i, c in enumerate(clients):
    c.accelerometer.notifications(lambda x, y=i: accelHandler(x, y))
    c.gyroscope.notifications(lambda x, y=i: gyroHandler(x, y))

#for c in clients:
    #c.accelerometer.start()
    #c.accelerometer.toggle_sampling(True)
    #c.gyroscope.start()
    #c.gyroscope.toggle_sampling(True)

time.sleep(5)

print('Unsubscribing...')
for c in clients:
    c.accelerometer.notifications(None)
    c.gyroscope.notifications(None)
    #c.accelerometer.toggle_sampling(False)
    #c.accelerometer.stop()
    #c.gyroscope.toggle_sampling(False)
    #c.gyroscope.stop()

time.sleep(5)

print('Disconnecting...')
for c in clients:
    c.disconnect()
