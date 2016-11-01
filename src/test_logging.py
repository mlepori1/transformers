# -*- coding: utf-8 -*-
"""
test_logging.py
  test mbientlab metawear logging functionality using python-wrapped C++ API

HISTORY
-------
2016-09-29: Created by Jonathan D. Jones
"""

import time

from pymetawear import libmetawear
from pymetawear.mbientlab.metawear.core import *
from pymetawear.mbientlab.metawear.sensor import AccelerometerBosch, AccelerometerBmi160
# (The CPRO devices use the Bosch BMI160 IMU, BMM150 magnetometer, BMP280 pressure + temp)

#from pymetawear_src.mbientlab.metawear.core import *
#from pymetawear_src.mbientlab.metawear.functions import setup_libmetawear

import copy

try:
    from pymetawear.backends.pybluez import PyBluezBackend
except ImportError as e:
    PyBluezBackend = e

#from ctypes import byref
#import os


# Exception classes
class PyMetaWearException(Exception):
    """ Main exception. """
    pass


class PyMetaWearConnectionTimeout(PyMetaWearException):
    """ Connection could not be established in time. """
    pass


class MetawearDevice:
    
    # Load the metawear C++ library
    #libmetawear = CDLL('pymetawear_src/libmetawear.so')
    #setup_libmetawear(libmetawear)
    
    def __init__(self, address, interface='hci0', timeout=None):
        
        self.address = address
        
        self.prev_time = -1
        self.data_time_offsets = []
        self.logged_data = []
        
        self.logger_id = None
        
        # Handling of timeout.
        timeout = None
        
        # Setup BLE communication backend and wait for board to initialize
        if isinstance(PyBluezBackend, Exception):
            raise PyMetaWearException(
                "pybluez[ble] package error: {0}".format(PyBluezBackend))
        
        self.backend = PyBluezBackend(self.address, interface=interface, timeout=timeout, debug=False)
        
        while (not self.backend.initialized) and (not
                libmetawear.mbl_mw_metawearboard_is_initialized(self.board)):
            self.backend.sleep(0.1)

        # Check if initialization has been completed successfully.
        if self.backend.initialized != Status.OK:
            if self.backend._initialization_status == Status.ERROR_TIMEOUT:
                raise PyMetaWearConnectionTimeout("libmetawear initialization status 16: Timeout")
            else:
                raise PyMetaWearException("libmetawear initialization status {0}".format(
                    self.backend._initialization_status))
        
        # Set up logging
        progress_update = Fn_Uint_Uint(self.progress_update_handler)
        unknown_entry = Fn_Ubyte_LongLong_ByteArray(self.unknown_entry_handler)
        self.download_handler = LogDownloadHandler(received_progress_update=progress_update,
                received_unknown_entry=unknown_entry,
                received_unhandled_entry=cast(None, Fn_DataPtr))
                
        self.logger_ready = Fn_VoidPtr(self.logger_ready_handler)
        
        self.cartesian_float_data = Fn_DataPtr(self.cartesian_float_data_handler)
        self.data_printer = Fn_DataPtr(self.data_printer_handler)


    @property
    def backend(self):
        """
        The backend object for this client.

        :return: The connected BLE backend.
        :rtype: :class:`pymetawear.backend.BLECommunicationBackend`
        """
        return self.backend


    @property
    def board(self):
        return self.backend.board
        
        
    def disconnect(self):
        """Disconnects this client from the MetaWear board."""
        libmetawear.mbl_mw_metawearboard_tear_down(self.board)
        libmetawear.mbl_mw_metawearboard_free(self.board)
        self.backend.disconnect()
    
    
    def test_acc_data(self):

        acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.board)
        #libmetawear.mbl_mw_datasignal_subscribe(acc_signal, self.data_printer)
        
        libmetawear.mbl_mw_acc_bosch_set_range(self.board, AccelerometerBosch.FSR_2G)
        libmetawear.mbl_mw_acc_set_odr(self.board, AccelerometerBmi160.ODR_50HZ)
        libmetawear.mbl_mw_acc_write_acceleration_config(self.board)
        
        # Create an MblMwDataLogger object by calling mbl_mw_datasignal_log
        # with the data signal you want to log. If successful, the callback
        # function will be executed with a MblMwDataLogger pointer. If
        # creating the logger failed, a null pointer will be returned.
        
        libmetawear.mbl_mw_datasignal_log(acc_signal, self.logger_ready)
        
        time.sleep(1)
        
        print('logger id: {}'.format(self.logger_id))
        logger = libmetawear.mbl_mw_logger_lookup_id(self.board, self.logger_id)
        libmetawear.mbl_mw_logger_subscribe(logger, self.cartesian_float_data)
        
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.board)
        libmetawear.mbl_mw_acc_start(self.board)
        libmetawear.mbl_mw_logging_start(self.board, 0)
        
        time.sleep(5)
        
        libmetawear.mbl_mw_logging_download(self.board, 10, byref(self.download_handler))
        
        time.sleep(5)
        
        libmetawear.mbl_mw_logging_stop(self.board)
        libmetawear.mbl_mw_acc_stop(self.board)
        libmetawear.mbl_mw_acc_disable_acceleration_sampling(self.board)

        num_samples = len(self.logged_data)
        time_interval = self.logged_data[-1][0] - self.logged_data[0][0]
        print('{} samples spanning {} milliseconds'.format(num_samples, time_interval))
    
    
    def progress_update_handler(self, entries_left, total_entries):
        print('{} / {}'.format(entries_left, total_entries))
    
    
    def unknown_entry_handler(self, ID, epoch, data, length):
        print('{} | received unknown log entry: id = {}'.format(epoch, ID))
    
    
    def logger_ready_handler(self, logger):
        
        if logger:
            print('logger ready')
        else:
            print('failed to create logger')
        
        self.logger_id = libmetawear.mbl_mw_logger_get_id(logger)
    
    
    def cartesian_float_data_handler(self, data):
        if (self.prev_time == -1):
            self.prev_time = data.contents.epoch
        else:
            self.data_time_offsets.append(data.contents.epoch - self.prev_time)
            self.prev_time = data.contents.epoch

        contents = copy.deepcopy(cast(data.contents.value, POINTER(CartesianFloat)).contents)
        self.logged_data.append((data.contents.epoch, contents.x, contents.y, contents.z))
        
        print('{} | ({} {} {})'.format(data.contents.epoch, contents.x, contents.y, contents.z))
    
    
    def data_printer_handler(self, data):
        print(data.contents.epoch)
    

if __name__ == '__main__':
    
    addresses = ('D6:B3:DA:FD:2E:DE',
                 'FC:63:6C:B3:4C:F6',
                 'F7:A1:FC:73:DD:23',
                 'C7:7D:36:B1:5E:7D')
    address = addresses[2]
    
    print('Connecting to {}...'.format(address))
    mw = MetawearDevice(address)
    
    mw.test_acc_data()
    
    print('Disconnecting...')
    mw.disconnect()
    
    