# -*- coding: utf-8 -*-
"""
libimgproc.py
  Library of functions for processing image frames recorded during block
  study trials

AUTHOR
  Jonathan D. Jones
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import subprocess

from duplocorpus import DuploCorpus


def drawKeyPoints(frame):
    """
    Use a simple blob detector to estimate block locations and poses in a given
    BGR video frame
    
    Args:
    -----
    [np array] frame: h-by-w-by-3 numpy array representing a BGR video frame
    
    Returns:
    --------
    [np array] frame:
    """
    
    #"""
    # Set blob detector parameters. Since DUPLO blocks are simple primary
    # colors, we'll look for blobs in the individual B, G, R channels
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    #params.filterByInertia = False
    #params.filterByConvexity = False
    #params.minArea = 50
    #params.maxArea = 300
    #"""
    
    #blob_detector = cv2.SimpleBlobDetector(params)
    
    """
    maxCorners = 20         # Don't return more corners than this
    qualityLevel = 0.05     # Max acceptable difference in quality from best corner
    minDistance = 5        # Minimum Euclidean distance between corners (in pixels)
    
    num_channels = frame.shape[2]
    for channel in range(num_channels):
        
        #blobs = blob_detector.detect(frame[:,:,channel])
        corners = cv2.goodFeaturesToTrack(frame[:,:,channel], maxCorners, qualityLevel, minDistance)
        corners = corners.astype(int)

        color = tuple(255 if j == channel else 0 for j in range(num_channels))
        radius = 2  # pixels
        for i in range(corners.shape[0]):
            corner_coords = tuple(corners[i][0])
            cv2.circle(frame, corner_coords, radius, color, -1)
    """
        
    lower_thresh = 250
    ratio = 2
    upper_thresh = lower_thresh * ratio
    
    ksize = (2,2)
    
    blurred = cv2.blur(frame, ksize)
    edge_frame = cv2.Canny(blurred, upper_thresh, lower_thresh)
    
    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, heirarchy = cv2.findContours(edge_frame, mode, method)
    
    contour_rects = []
    poly_contours = []
    eps = 3
    for contour_idx, contour in enumerate(contours):
        contour_pts = contour[:,0,:]
        
        rect = cv2.minAreaRect(contour_pts)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        contour_rects.append(box)
        
        poly_contour = cv2.approxPolyDP(contour_pts, eps, True)
        poly_contours.append(poly_contour)
    
    contour_frame = np.zeros(frame.shape)
    for contour_idx, contour in enumerate(poly_contours):
        contour_pts = contours[contour_idx][:,0,:]
        
        # Determine closest color from interior of contour
        max_row = contour_pts[:,1].max()
        min_row = contour_pts[:,1].min()
        max_col = contour_pts[:,0].max()
        min_col = contour_pts[:,0].min()
        colors = frame[min_row:max_row, min_col:max_col, :]
        avg_color = np.mean(np.mean(colors, axis=0), axis=0)
        max_color = avg_color.argmax()
        color = np.zeros(avg_color.shape)
        color[max_color] = 255
        
        cv2.drawContours(contour_frame, poly_contours, contour_idx, color)
    
    """
    edges = edge_frame == 255
    
    labeled_frame = np.zeros(frame.shape)
    num_channels = frame.shape[2]
    for i in range(num_channels):
        channel = frame[:,:,i]
        channel[edges] = 0
        labeled_frame[:,:,i] = channel
    """
    
    return contour_frame.astype('uint8')


def detectBlocks(frame):
    """
    Rectangle detection using the Hough transform.
    """
    
    


if __name__ == '__main__':
    
    c = DuploCorpus()
    
    rgb_frame_fns = c.getRgbFrameFns(1)
    
    for file_path in rgb_frame_fns[50:51]: #[250:255]:
        frame = cv2.imread(file_path)
        labeled_frame = drawKeyPoints(frame)
        
        #_, fn = os.path.split(file_path)
        #cv2.imwrite(os.path.join(out_path, fn), labeled_frame)
        plt.imshow(labeled_frame[:,:,[2, 1, 0]])
    
    """
    frame_fmt = os.path.join(out_path, '%6d.png')
    video_path = os.path.join('output', 'rgb', trial_id + '.avi')
    make_video = ['ffmpeg', '-f', 'image2', '-i', frame_fmt, '-r', '30', video_path]
    subprocess.call(make_video)
    """