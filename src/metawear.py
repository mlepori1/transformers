
from pymetawear.client import *
import time
import numpy as np
from matplotlib import pyplot as plt

class DataStreamingApp:
    
    def __init__(self, accel_rate=25, accel_range=8, gyro_rate=25, gyro_range=1000):
        
        self.clients = []
        
        self.accel_data = []
        self.gyro_data = []
        
        self.accel_rate = accel_rate
        self.accel_range = accel_range
        self.gyro_rate = gyro_rate
        self.gyro_range = gyro_range
        

    def accelHandler(self, data, client_index):
        #source_str = "CLIENT {} | A | ".format(client_index)
        #data_str = str(data)
        #print(source_str + data_str)
        
        self.accel_data[client_index].append(data)
        
    
    def gyroHandler(self, data, client_index):
        #source_str = "CLIENT {} | G | ".format(client_index)
        #data_str = str(data)
        #print(source_str + data_str)
        
        self.gyro_data[client_index].append(data)
    
    
    def connect(self, addresses, stream_high_freq=False):
        for i, address in enumerate(addresses):
            print('Connecting to {} as client {}...'.format(address, i))
            self.clients.append(MetaWearClient(address, backend='pybluez'))
            self.accel_data.append([])
            self.gyro_data.append([])
            
        for i, client in enumerate(self.clients):
            client.accelerometer.set_settings(data_rate=self.accel_rate,
                                              data_range=self.accel_range)
            client.accelerometer.high_frequency_stream = stream_high_freq
            client.accelerometer.notifications(lambda x, y=i: self.accelHandler(x, y))
            
            client.gyroscope.set_settings(data_rate=self.gyro_rate,
                                          data_range=self.gyro_range)
            client.gyroscope.high_frequency_stream = stream_high_freq
            client.gyroscope.notifications(lambda x, y=i: self.gyroHandler(x, y))
            print('Subscribing to client {}...'.format(i))
    
    
    def startStream(self):
        for client in self.clients:
            client.accelerometer.start()
            client.accelerometer.toggle_sampling(True)
            client.gyroscope.start()
            client.gyroscope.toggle_sampling(True)
    
    
    def stopStream(self):
        for i, client in enumerate(self.clients):
            print('Unsubscribing from client {}...'.format(i))
            client.accelerometer.notifications(None)
            client.gyroscope.notifications(None)
            client.accelerometer.toggle_sampling(False)
            client.accelerometer.stop()
            client.gyroscope.toggle_sampling(False)
            client.gyroscope.stop()
    
    
    def disconnect(self):
        for i, client in enumerate(self.clients):
            print('Disconnecting from client {}...'.format(i))
            client.disconnect()
    
    
    def reset(self):
        for client in self.clients:
            client.soft_reset()
    
    
    def plotData(self):
        
        for i, data in enumerate(self.accel_data):
            times = np.array([x[0] for x in data], dtype=float)
            times -= times[0]
            times /= 1000.0
            
            num_samples = times.size
            sample_freq = num_samples / times[-1]
            
            samples = np.array([x[1] for x in data])
            sample_norms = (samples ** 2).sum(axis=1) ** 0.5
            
            f, axes = plt.subplots(2, sharex=True)
            axes[0].scatter(times, sample_norms, c='r')
            axes[0].plot(times, sample_norms)
            axes[0].set_ylabel('Acceleration magnitude')
            title_str = '{} samples received from client {} at {:.1f} Hz'
            axes[0].set_title(title_str.format(num_samples, i, sample_freq))
            axes[1].plot(times[1:], np.diff(times))
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('dt (seconds)')
        
        for i, data in enumerate(self.gyro_data):
            times = np.array([x[0] for x in data], dtype=float)
            times -= times[0]
            times /= 1000.0
            
            num_samples = times.size
            sample_freq = num_samples / times[-1]
            
            samples = np.array([x[1] for x in data])
            sample_norms = (samples ** 2).sum(axis=1) ** 0.5
            
            f, axes = plt.subplots(2, sharex=True)
            axes[0].scatter(times, sample_norms, c='r')
            axes[0].plot(times, sample_norms)
            axes[0].set_ylabel('Angular velocity magnitude')
            title_str = '{} samples received from client {} at {:.1f} Hz'
            axes[0].set_title(title_str.format(num_samples, i, sample_freq))
            axes[1].plot(times[1:], np.diff(times))
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('dt (seconds)')
        
        plt.show()
        


addresses = ('D6:B3:DA:FD:2E:DE',
             'FC:63:6C:B3:4C:F6',
             'F7:A1:FC:73:DD:23',
             'C7:7D:36:B1:5E:7D')
addresses = addresses[1:3]

app = DataStreamingApp(accel_rate=25, gyro_rate=25)
app.connect(addresses, stream_high_freq=True)
app.startStream()

stream_time = 60 * 3
time.sleep(stream_time)

app.stopStream()

time.sleep(1)

app.disconnect()

time.sleep(1)

app.plotData()

