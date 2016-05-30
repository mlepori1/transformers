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
    
    ksize = (3,3)
    blurred = cv2.blur(frame, ksize)
    plt.imshow(blurred[:,:,[2, 1, 0]])
    plt.show()
    
    lower_thresh = 200
    ratio = 2
    upper_thresh = lower_thresh * ratio
    edge_frame = cv2.Canny(blurred, upper_thresh, lower_thresh)
    plt.imshow(edge_frame, cmap='gray')
    plt.show()
    
    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, heirarchy = cv2.findContours(edge_frame, mode, method)
    
    """
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
    """
    
    contour_frame = np.zeros(frame.shape)
    for contour_idx, contour in enumerate(contours):
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
        
        cv2.drawContours(contour_frame, contours, contour_idx, color)
    
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
    
    trial_id = 4
    
    c = DuploCorpus()
    
    rgb_frame_fns = c.getRgbFrameFns(trial_id)
    depth_frame_fns = c.getDepthFrameFns(trial_id)
    
    for rgb_path, depth_path in zip(rgb_frame_fns, depth_frame_fns):
        
        # Read in frames and plot
        rgb_frame = cv2.imread(rgb_path)
        rgb_frame = rgb_frame[:,:,[2, 1, 0]] # BGR to RGB
        #plt.imshow(rgb_frame)
        #plt.show()
        depth_frame = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype('uint8')
        #plt.imshow(depth_frame, cmap=plt.get_cmap('gray'))
        #plt.show()
        
        depth_frame[depth_frame == 0] = int(np.median(depth_frame))
        #plt.imshow(depth_frame, cmap=plt.get_cmap('gray'))
        #plt.show()
        
        # Get colors from rgb data
        yellow_channel = rgb_frame[:,:,0:2].sum(axis=2) / 2.0
        gray_channel = rgb_frame.sum(axis=2) / 3.0
        
        # Components along subspaces orthogonal to r/g/b/y/gray color subspaces
        zero = np.zeros(rgb_frame[:,:,0].shape)
        orth_red = rgb_frame - np.dstack((rgb_frame[:,:,0], zero, zero))
        orth_green = rgb_frame - np.dstack((zero, rgb_frame[:,:,1], zero))
        orth_blue = rgb_frame - np.dstack((zero, zero, rgb_frame[:,:,2]))
        orth_yellow = rgb_frame - np.dstack((yellow_channel, yellow_channel, zero))
        orth_gray = rgb_frame - np.dstack((gray_channel, gray_channel, gray_channel))
        
        # Distance to r/g/b/y/gray subspaces for each pixel (euclidean norm
        # in color space)
        dist_red = (orth_red ** 2).sum(axis=2) ** 0.5
        dist_green = (orth_green ** 2).sum(axis=2) ** 0.5
        dist_blue = (orth_blue ** 2).sum(axis=2) ** 0.5
        dist_yellow = (orth_yellow ** 2).sum(axis=2) ** 0.5
        dist_gray = (orth_gray ** 2).sum(axis=2) ** 0.5
        
        extended = np.dstack((dist_red, dist_green, dist_blue, dist_yellow,
                              dist_gray))
        min_index = extended.argmin(axis=2)
        
        color_frame = np.zeros(rgb_frame.shape)
        color_frame[min_index == 0,:] = np.array([255, 0, 0])
        color_frame[min_index == 1,:] = np.array([0, 255, 0])
        color_frame[min_index == 2,:] = np.array([0, 0, 255])
        color_frame[min_index == 3,:] = np.array([255, 255, 0])
        color_frame[min_index == 4,:] = np.array([128, 128, 128])
        color_frame = color_frame.astype('uint8')
        
        #plt.imshow(color_frame)
        #plt.show()
        
        # Get edge contours from depth data and draw onto color image
        lower_thresh = 50
        ratio = 1.5
        upper_thresh = lower_thresh * ratio
        edge_frame = cv2.Canny(depth_frame, upper_thresh, lower_thresh)
        #plt.imshow(edge_frame, cmap='gray')
        #plt.show()
        
        # Detect contours
        mode = cv2.RETR_TREE
        method = cv2.CHAIN_APPROX_SIMPLE
        _, contours, _ = cv2.findContours(edge_frame, mode, method)
        contours = [cont for cont in contours if cont.shape[0] > 20]
        contour_frame = rgb_frame.copy()
        for i, contour in enumerate(contours):
            color = (0, 0, 0)
            cv2.drawContours(color_frame, contours, i, color)
        #plt.imshow(contour_frame)
        #plt.show()
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edge_frame, 50, 2 * np.pi / 180, 50, 50, 5)
        line_frame = rgb_frame.copy()
        line_frame[edge_frame.astype(bool),:] = np.zeros(3, dtype='uint8')
        """
        for i in range(line_frame.shape[2]):
            channel = line_frame[:,:,i]
            channel[edge_frame.astype(bool)] = 0
            line_frame[:,:,i] = channel
        """
        for l in lines:
            points = l[0]
            intensity = 0
            color = (0, 255, 0)
            cv2.line(line_frame, (points[0], points[1]), (points[2], points[3]), color)
        #plt.imshow(line_frame)
        #plt.show()
        
        """
        # Calculate and plot color space representation
        # Unravel along dimension 0 and 1
        f = frame[0::5,0::5,:]
        t = tuple(f[i,j,:] for i in range(f.shape[0]) for j in range(f.shape[1]))
        X = np.vstack(t)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2])
        plt.show()
        
        width = 3
        kernel = np.ones((width,width),np.uint8)
        closed = cv2.morphologyEx(color_frame, cv2.MORPH_CLOSE, kernel)
        plt.imshow(closed)
        plt.show()
        plt.imshow(color_frame)
        plt.show()
        """
        
        """
        labeled_frame = drawKeyPoints(color_frame[:,:,[2, 1, 0]])
        plt.imshow(labeled_frame[:,:,[2, 1, 0]])
        plt.show()
        """
        
        orig_frames = np.hstack((rgb_frame, np.dstack(3 * (depth_frame,))))
        processed_frames = np.hstack((color_frame, np.dstack(3 * (edge_frame,))))
        meta_frame = np.vstack((orig_frames, processed_frames))
        _, fn = os.path.split(rgb_path)
        cv2.imwrite(os.path.join(c.paths['working'], fn), meta_frame[:,:,[2, 1, 0]])
    
    #"""
    import platform
    av_util = ''
    if platform.system() == 'Linux':
        av_util = 'avconv'
    elif platform.system() == 'Darwin':
        av_util = 'ffmpeg'
    frame_fmt = os.path.join(c.paths['working'], '%6d.png')
    video_path = os.path.join(c.paths['working'], '{}.avi'.format(trial_id))
    make_video = [av_util, '-y', '-f', 'image2', '-i', frame_fmt, '-c:v',
                  'libx264', '-r', '30', '-pix_fmt', 'yuv420p', video_path]
    subprocess.call(make_video)
    #"""