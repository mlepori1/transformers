"""
capture_video.py
  demo script to save webcam frames using openCV

AUTHOR
  Jonathan D. Jones
"""

import cv2
import time
import os

dirname = str(int(time.time()))
os.mkdir(dirname)

cap = cv2.VideoCapture(0)

i = 0
while i < 600:
    ret, frame = cap.read()
    filename = str(i) + ".png"
    cv2.imwrite(os.path.join(dirname, filename), frame)
    i += 1

cap.release()
