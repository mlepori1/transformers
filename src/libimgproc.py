# -*- coding: utf-8 -*-
"""
libimgproc.py
  Library of functions for processing image frames recorded during block
  study trials

AUTHOR
  Jonathan D. Jones
"""

import os
import subprocess

import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage import feature
from skimage import transform

from duplocorpus import DuploCorpus


def quantizeColors(rgb_image):
    """
    Apply color quantization by assigning each pixel to the nearest color
    subspace in RGB space.
    
    Args:
    -----
    [cv mat] rgb_image: h-by-w-by-3 image array with third dimension
      representing the intensities of the red, green, and blue channels
    
    Returns:
    --------
    [cv mat] quantized_image: h-by-w-by-5 image array with third dimension
      representing the quantization responses in the red, green, blue, yellow,
      and gray color subspaces
    """
    
    # Red, green, blue, yellow, gray
    rgb_values = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (128, 128, 128)]
    
    yellow_channel = rgb_frame[:,:,0:2].sum(axis=2) / 2.0
    gray_channel = rgb_frame.sum(axis=2) / 3.0
    
    # Components along orthogonal complement of r/g/b/y/gray color subspaces
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
    
    channel_responses = np.zeros(extended.shape, dtype=bool)
    quantized_image = np.zeros(rgb_image.shape, dtype='uint8')
    for i, value in enumerate(rgb_values):
        in_region = min_index == i
        channel_responses[:,:,i] = in_region
        quantized_image[in_region, :] = np.array(value)
    
    return quantized_image, channel_responses


def quantizeDepths(depth_image):
    """
    """
    
    thresholds = (0, 50, 65, 85, 256)
    
    # Any pixel intensity actually equal to zero is either directly in front
    # of the camera (unlikely) or was thrown out when the camera registered
    # this frame to the RGB camera (more likely). By setting all 0 values near
    # the maximum (ie far away), we treat them as part of the background.
    depth_image[depth_image == 0] = depth_image.max() - 1
    
    # Assemble quantized image array with one channel for each depth region
    channel_responses = np.zeros(depth_image.shape + (len(thresholds) - 1,),
                               dtype=bool)
    quantized_image = np.zeros(depth_image.shape, dtype='uint8')
    for i in range(len(thresholds) - 1):
        
        # Quantize pixels in this depth region by assigning them to the mean
        # value
        lower_thresh = thresholds[i]
        upper_thresh = thresholds[i+1]
        value = (lower_thresh + upper_thresh) / 2
        
        in_region = np.logical_and(depth_image >= lower_thresh,
                                   depth_image <  upper_thresh)
        channel_responses[:,:,i] = in_region
        quantized_image[in_region] = value
    
    return quantized_image, channel_responses


if __name__ == '__main__':
    
    trial_id = 4
    
    c = DuploCorpus()
        
    rgb_frame_fns = c.getRgbFrameFns(trial_id)[50:51]
    depth_frame_fns = c.getDepthFrameFns(trial_id)[50:51]
    
    for rgb_path, depth_path in zip(rgb_frame_fns, depth_frame_fns):
        
        # Read in frames and plot
        rgb_frame = io.imread(rgb_path)
        depth_frame = io.imread(depth_path).astype('uint8')
        
        c_img, c_responses = quantizeColors(rgb_frame)
        d_img, d_responses = quantizeDepths(depth_frame)
        
        """
        orig_frames = np.hstack((rgb_frame, np.dstack(3 * (depth_frame,))))
        processed_frames = np.hstack((c_img, np.dstack(3 * (d_img,))))
        meta_frame = np.vstack((orig_frames, processed_frames))
        _, fn = os.path.split(rgb_path)
        cv2.imwrite(os.path.join(c.paths['working'], fn), meta_frame[:,:,])
        """
        
        layers = []
        for i in range(d_responses.shape[2]):
            
            """
            # Select a depth layer
            layer = np.zeros(c_img.shape, dtype='uint8')
            layer[d_responses[:,:,i],:] = c_img[d_responses[:,:,i],:]
            
            """
            
            layer = feature.canny(d_responses[:200,:300,i].astype('uint8') * 255)
            rows, cols = layer.shape
            
            f = plt.figure()
            plt.imshow(layer, cmap=plt.cm.gray)
            
            """
            # Detect lines via Hough transform
            h, theta, d = transform.hough_line(layer)
            _, angles, dists = transform.hough_line_peaks(h, theta, d, )
            
            # Plot lines
            for angle, dist in zip(angles, dists):
                y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
                plt.plot((0, cols), (y0, y1), '-r')
            plt.axis((0, cols, rows, 0))
            layers.append(layer)
            """
            
            lines = transform.probabilistic_hough_line(layer, threshold=20,
                                                       line_length=10,
                                                       line_gap=50)
            for line in lines:
                pt1, pt2 = line
                plt.plot((pt1[0], pt2[0]), (pt1[1], pt2[1]), '-r')
            plt.axis((0, cols, rows, 0))
            layers.append(layer)
        
        layer_frame = np.vstack((np.hstack(tuple(layers[0:2])),
                                 np.hstack(tuple(layers[2:4]))))
        plt.figure()
        plt.imshow(layer_frame, cmap=plt.cm.gray); plt.show()
        _, fn = os.path.split(rgb_path)
        io.imsave(os.path.join(c.paths['working'], fn), layer_frame)
        
    
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