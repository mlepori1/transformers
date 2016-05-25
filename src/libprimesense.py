# -*- coding: utf-8 -*-
"""
libprimesense
  Library for streaming RGBD data from a primesense camera

AUTHOR
  Jonathan D. Jones

NOTE
  Depth frames look blank, but they can be opened in openCV with
  depth_img = cv2.imread([filename], cv2.IMREAD_ANYDEPTH)
"""

import os
import csv
import time

import numpy as np
import cv2

from primesense import openni2


def stream(frame_base_path, timestamp_path, img_types, die, q):
    """
    Stream data from camera until die is set

    Args:
    -----
    [str] frame_base_path:
    [str] timestamp_path:
    [tuple(str)] img_types:
    [mp event] die:
    [mp queue] q:
    """
    
    # Create directory for video frames
    for img_type in img_types:
        frame_dir = '{}-{}'.format(frame_base_path, img_type)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
    
    # Set up file for writing timestamps
    f_timestamp = open(timestamp_path, 'w')
    timestamp_writer = csv.writer(f_timestamp)
    
    # Open video streams
    print("Opening camera...")
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    rgb_stream = dev.create_color_stream()
    
    frame_index = 0
    
    print("Starting RGBD streams...")
    if 'depth' in img_types:
        depth_stream.start()
    if 'rgb' in img_types:
        rgb_stream.start()
    while not die.is_set():
        
        for img_type in img_types:
            
            frametime = time.time()
            
            # Read frame and convert to image array (either color or depth)
            if img_type == 'depth':
                frame = depth_stream.read_frame()
                data = frame.get_buffer_as_uint16()
                img_array = np.ndarray((frame.height, frame.width),
                                        dtype=np.uint16, buffer=data)
            elif img_type == 'rgb':
                frame = rgb_stream.read_frame()
                data = frame.get_buffer_as_uint8()
                img_array = np.ndarray((frame.height, 3*frame.width),
                                       dtype=np.uint8, buffer=data)
                img_array = np.dstack((img_array[:,2::3], img_array[:,1::3],
                                       img_array[:,0::3]))
            
            # Write video frame to file and record timestamp
            fn = '{:06d}.png'.format(frame_index)
            path = os.path.join('{}-{}'.format(frame_base_path, img_type), fn)
            cv2.imwrite(path, img_array)
            timestamp_writer.writerow((frametime, 0, frame_index, img_type))
            
            # Put
            if img_type == 'rgb' and q.empty():
                q.put(path)

        frame_index += 1

    # Stop streaming
    print("Closing camera")
    if 'depth' in img_types:
        depth_stream.stop()
    if 'rgb' in img_types:
        rgb_stream.stop()
    openni2.unload()
    
    f_timestamp.close()

