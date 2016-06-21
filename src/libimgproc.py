# -*- coding: utf-8 -*-
"""
libimgproc.py
  Library of functions for processing image frames recorded during block
  study trials

AUTHOR
  Jonathan D. Jones
"""

import csv
import os
import subprocess

import numpy as np
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import linalg as la

from skimage import io
from skimage import feature
from skimage import transform
from skimage import measure
from skimage import morphology
from skimage import draw
from skimage import color

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
    thresholds = (0, 559, 577, 594)
    
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


def getFocalLength():
    """
    Perform camera calibration under the assumption of a pinhole camera.    
    
    Returns:
    --------
    [float] f: Camera focal length in metric units
    """
    
    # (Camera calibration)
    block_1_corners = np.array([[120, 101], [119, 87], [133, 85], [134, 100]])
    block_2_corners = np.array([[134, 100], [133, 85], [147, 84], [148, 99]])
    block_3_corners = np.array([[149, 97], [148, 83], [161, 82], [162, 99]])
    block_4_corners = np.array([[164, 96], [163, 81], [176, 80], [177, 95]])
    side_lens = np.zeros(block_1_corners.shape[0])
    side_lens[:-1] = (np.diff(block_1_corners, axis=0) ** 2).sum(axis=1) ** 0.5
    side_lens[-1] = ((block_1_corners[-1,:] - block_1_corners[0,:]) ** 2).sum() ** 0.5
    z_m = 600   # mm
    
    # Side length of a 2x2 DUPLO brick (from stack excange link below)
    # http://bricks.stackexchange.com/questions/1588/
    #   what-are-the-exact-dimensions-of-a-duplo-brick
    block_width_metric = 31.8  # mm
    block_height_metric = 19.2 # mm
    # Derived length of a 4x2 rectangular DUPLO brick
    block_length_metric = 2 * block_width_metric
    
    # Average focal length assuming a pinhole camera
    f = (side_lens * (z_m - block_height_metric) / block_width_metric).mean() # pixels
    
    # Block corner locations in 3D assuming coordinate frames of metric space
    # and the image plane are aligned
    metric_corners = block_1_corners * (z_m - block_height_metric) / f
    
    # The depth camera directly records distance in mm
    # (end camera calibration)
    
    return f


def pixel2metric(block_dim, block_depth):
    """
    """
    
    f = getFocalLength()
    pixel_to_metric = block_depth / f   # mm / pixel
    metric_est = pixel_to_metric * block_dim
    
    return metric_est
    

def calcResidual(block_width, block_length, block_depth):
    """
    """
    
    block_width_metric = 31.8  # mm
    block_height_metric = 19.2 # mm
    block_length_metric = 2 * block_width_metric
    
    ratio = block_length / block_width
    if ratio < 1.5: # Midway between 1 (square) and 2 (rectangle)
        side_len = 0.5 * (block_length + block_width)
        metric_side_est = pixel2metric(side_len, block_depth)
        residual_w = abs(block_width_metric - metric_side_est) / block_width_metric
        residual_l = residual_w
    else:
        metric_width_est = pixel2metric(block_width, block_depth)
        metric_length_est = pixel2metric(block_length, block_depth)
        residual_w = abs(block_width_metric - metric_width_est) \
                   / block_width_metric
        residual_l = abs(block_length_metric - metric_length_est) \
                   / block_length_metric
    
    return residual_w, residual_l, ratio
    

def detectBlocks(rgb_image, depth_image):
    """
    """
    
    rows, cols = depth_image.shape
    #rgb_edges = feature.canny(color.rgb2gray(rgb_image))
    depth_edges = feature.canny(depth_image.astype('uint8'))
    c_img, c_responses = quantizeColors(rgb_frame)
    
    # Contours
    contours = measure.find_contours(depth_edges.astype('uint8'), 0.5,
                                     fully_connected='high')                
    contours = [c[:,::-1] for c in contours]
    contour_feats = np.zeros((len(contours), 11))
    for j, contour in enumerate(contours):
        
        interior_pts = np.column_stack(draw.polygon(contour[:,1], contour[:,0]))
        depth_pts = depth_frame[interior_pts[:,0], interior_pts[:,1]]
        block_depth = depth_pts.mean()       # mm
        
        contour_colors = c_responses[interior_pts[:,0],interior_pts[:,1],:]
        color_hist = contour_colors.sum(axis=0)
        
        #axes[0,1].plot(contour[:, 0], contour[:, 1], color=c_color)
        
        # Determine width and length of block by finding mean distance from
        # one extreme point to its nearest and second-nearest neighbor
        extreme_pts = [contour[contour[:,0].argmax(),:],
                       contour[contour[:,1].argmax(),:],
                       contour[contour[:,0].argmin(),:],
                       contour[contour[:,1].argmin(),:]]
        block_lengths = np.zeros(4)
        block_widths = np.zeros(4)
        for i in range(len(extreme_pts)):
            block_length = np.inf
            block_width = np.inf
            p0 = extreme_pts[i]
            for pt in extreme_pts[:i] + extreme_pts[i+1:]:
                dist = la.norm(p0 - pt)
                if dist < block_width:
                    block_length = block_width
                    block_width = dist
                elif dist < block_length:
                    block_length = dist
            block_lengths[i] = block_length
            block_widths[i] = block_width
        block_length = block_lengths.mean()
        block_width = block_widths.mean()
        block_dims = np.array([block_length, block_width])
        
        residual_w, residual_l, ratio = calcResidual(block_width, block_length,
                                                     block_depth)
        size_tol = 0.25  # allow max 25% deviation
        if  residual_w > size_tol or residual_l > size_tol:
            #print(residual_w, ' ', residual_l)
            continue
                
        mean = interior_pts.mean(axis=0)
        interior_centered = interior_pts - np.vstack(interior_pts.shape[0] * (mean,))
        U, S, V_T = la.svd(interior_centered)
        # This is a trick I saw on stack exchange:
        #   http://math.stackexchange.com/questions/301319/
        #   derive-a-rotation-from-a-2d-rotation-matrix
        # Except keep in mind that we have V TRANSPOSE, not V
        theta = np.rad2deg(np.arctan2(V_T[0,1], V_T[0,0]))
        
        """
        c_corners = np.array([[ block_length,  block_width],
                              [ block_length, -block_width],
                              [-block_length, -block_width],
                              [-block_length,  block_width]]) / 2
        c_corners = c_corners.dot(V_T) + np.vstack(c_corners.shape[0] * (mean,))
        """
        
        row = np.hstack((mean, np.array([theta]), block_dims,
                         np.array([block_depth]), color_hist))
        contour_feats[j,:] = row
    
    contour_feats = contour_feats[contour_feats[:,0] != 0, :]
    
    # Merge contour features with centroids close to each other (these are
    # probably multiple detections of the same object)
    min_dist = 1
    good_feats = np.zeros(contour_feats.shape)
    for i in range(contour_feats.shape[0]):
        centroid = contour_feats[i,0:2]
        for j in range(i):
            other_centroid = contour_feats[j,0:2]
            dist = la.norm(centroid - other_centroid)
            #print(dist)
            if dist < min_dist:
                good_feats[j,-5:] += contour_feats[i,-5:]
                break
        else:
            good_feats[i,:] = contour_feats[i,:]
    
    contour_feats = good_feats[good_feats[:,0] != 0, :]
    return contour_feats


if __name__ == '__main__':
        
    trial_id = 0
    
    """
    tracked_objs = []
    filtered_path = '/Users/jonathan/filtered_coords/'
    import glob
    for csv_fn in glob.glob(os.path.join(filtered_path, '*.csv')):
        M = np.loadtxt(csv_fn, delimiter=',')
        M[:,-5:] = np.cumsum(M[:,-5:], axis=0)
        tracked_objs.append(M)
    """
    
    c = DuploCorpus()
    i = 20
    rgb_frame_fns = c.getRgbFrameFns(trial_id)#[i:i+1]
    depth_frame_fns = c.getDepthFrameFns(trial_id)#[i:i+1]
    
    # Red, green, blue, yellow, gray (drawn as black)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 0, 0)]
    color_chars = np.array(['r', 'g', 'b', 'y', 'k'])
    
    plt.ioff()
    tracks = []
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_frame_fns, depth_frame_fns)):
        
        _, frame_fn = os.path.split(rgb_path)
        frame_num = os.path.splitext(frame_fn)[0]
        
        # Read in frames and plot
        rgb_frame = io.imread(rgb_path)
        depth_frame = io.imread(depth_path)
        
        #"""
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rgb_frame)
        axes[1].imshow(depth_frame, cmap=plt.cm.gray)
        #"""
        
        fn = '/Users/jonathan/block_coords/{}.csv'.format(frame_num)
        contour_feats = detectBlocks(rgb_frame, depth_frame)
        np.savetxt(fn, contour_feats, fmt='%.2f', delimiter=',')
        axes[0].scatter(contour_feats[:,1], contour_feats[:,0])
        
        if i == 0:
            # Initialize tracks
            for j in range(contour_feats.shape[0]):
                tracks.append(contour_feats[j,:])
        else:
            # Assign detections to tracks
            for t_idx, track in enumerate(tracks):
                track_centroid = track[0:2]
                nn_dist = np.Inf
                nn_idx = -1
                # Find the nearest detection
                # FIXME: don't let track centroids merge
                for j in range(contour_feats.shape[0]):
                    contour_centroid = contour_feats[j,0:2]
                    dist = la.norm(contour_centroid - track_centroid)
                    if dist < nn_dist:
                        nn_dist = dist
                        nn_idx = j
                # Update the nearest neighbor
                track[0:-5] = contour_feats[nn_idx,0:-5]
                track[-5:] += contour_feats[nn_idx,-5:]
                tracks[t_idx] = track
        
        #"""
        filtered_centroids = tuple(x[0:2] for x in tracks)
        filtered_centroids = np.vstack(filtered_centroids)
        color_hists = tuple(x[-5:] for x in tracks)
        color_hists = np.vstack(color_hists)
        colors = color_chars[color_hists[:,:-1].argmax(axis=1)]
        axes[1].scatter(filtered_centroids[:,1], filtered_centroids[:,0], c=colors)
        
        for ax in axes.ravel():
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
        _, fn = os.path.split(rgb_path)
        plt.savefig(os.path.join(c.paths['working'], fn), format='png')
        plt.close()
        #"""
        
        
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
    
    