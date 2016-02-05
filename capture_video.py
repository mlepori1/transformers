"""
capture_video.py
  demo script to save webcam frames using openCV

AUTHOR
  Jonathan D. Jones
"""

from primesense import openni2
import time
import os
import numpy as np
import cv2

dirname = str(int(time.time()))
os.mkdir(dirname)

openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_sensor_info(openni2.SENSOR_DEPTH))

color_stream = dev.create_color_stream()
color_stream.start()

depth_stream = dev.create_depth_stream()
depth_stream.start()

i = 0
max_iters = 60
while i < max_iters:

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_array = np.ndarray((depth_frame.height, depth_frame.width),
                             dtype=np.uint16, buffer=depth_data)
    filename = str(i) + "_depth" + ".png"
    cv2.imwrite(os.path.join(dirname, filename), depth_array)

    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()
    color_array = np.ndarray((color_frame.height, 3*color_frame.width),
                             dtype=np.uint8, buffer=color_data)
    color_array = np.dstack((color_array[:,2::3], color_array[:,1::3],
        color_array[:,0::3]))
    filename = str(i) + "_rgb"  + ".png"
    cv2.imwrite(os.path.join(dirname, filename), color_array)

    i += 1

depth_stream.stop()
color_stream.stop()
openni2.unload()
