# -*- coding: utf-8 -*-
"""
libprimesense
  Library for interfacing with a primesense RGBD camera

AUTHOR
  Jonathan D. Jones
"""

import os
import csv
import time

import numpy as np
import cv2

from primesense import openni2


def stream(frame_path, timestamp_path, die, q):
    """
    Stream data from camera until die is set

    Args:
    -----
    [dict(str->cv stream)] dev_name:
    [str] path: Path (full or relative) to image output directory
    """
    
    # Create directory for rgb frames
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    # Set up file for writing timestamps
    f_timestamp = open(timestamp_path, 'w')
    timestamp_writer = csv.writer(f_timestamp)
    
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
        img_array = np.dstack((img_array[:,2::3], img_array[:,1::3],
                               img_array[:,0::3]))
        
        # Write to file
        fn = '{:06d}.png'.format(frame_index)
        path = os.path.join(frame_path, fn)
        cv2.imwrite(path, img_array)
        timestamp_writer.writerow((frametime, 0, frame_index, 'rgb'))
        
        if q.empty():
            q.put(path)

        frame_index += 1

    # Stop streaming
    print("Closing RGB camera")    
    stream.stop()
    openni2.unload()
    
    f_timestamp.close()

