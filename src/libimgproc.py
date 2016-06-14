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


def calibrateCamera(p_pixel, z_m):
    """
    Perform camera calibration under the assumption of a pinhole camera.
    
    Args:
    -----
    [np array] p_pixel: Block corner coordinates in pixels
    [float] z_m: Block's distance from camera
    
    Returns:
    --------
    [np array] p_metric:
    [np vector] c: Camera coordinates in metric units
    [float] f: Camera focal length in metric units
    """
    
    return


if __name__ == '__main__':
    
    #plt.ioff()
    
    trial_id = 5
    
    c = DuploCorpus()
        
    rgb_frame_fns = c.getRgbFrameFns(trial_id)[50:51]
    depth_frame_fns = c.getDepthFrameFns(trial_id)[50:51]
    
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
    
    # It looks like a model for the depth camera is:
    #   depth = max_depth - intensity
    # where max_depth is about 636
    depth_offset = 636
    # (end camera calibration)
    
    for rgb_path, depth_path in zip(rgb_frame_fns, depth_frame_fns):
        
        _, frame_fn = os.path.split(rgb_path)
        frame_num = int(os.path.splitext(frame_fn)[0])
        
        # Read in frames and plot
        rgb_frame = io.imread(rgb_path)
        #plt.imshow(rgb_frame); #plt.show()
        depth_frame = io.imread(depth_path).astype('uint8') #[25:200,25:300]
        #plt.imshow(depth_frame, cmap=plt.cm.gray) #; plt.show()
        
        c_img, c_responses = quantizeColors(rgb_frame)
        d_img, d_responses = quantizeDepths(depth_frame)
        
        """
        orig_frames = np.hstack((rgb_frame, np.dstack(3 * (depth_frame,))))
        processed_frames = np.hstack((c_img, np.dstack(3 * (d_img,))))
        meta_frame = np.vstack((orig_frames, processed_frames))
        _, fn = os.path.split(rgb_path)
        cv2.imwrite(os.path.join(c.paths['working'], fn), meta_frame[:,:,])
        """
        
        """
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rgb_frame)
        axes[1].imshow(depth_offset - depth_frame, cmap=plt.cm.gray)
        """
                
        layers = []
        for i in range(d_responses.shape[2] - 1):
            
            #plt.figure()
            #plt.imshow(rgb_frame)
                
            
            """
            plt.figure()
            plt.imshow(rgb_frame)
            """
            
            # Select a depth layer
            #layer = np.zeros(c_img.shape, dtype='uint8')
            #layer[d_responses[:,:,i],:] = c_img[d_responses[:,:,i],:]
            
            layer = d_responses[:,:,i].astype('uint8')
            layer[200:,:] = 0
            layer[:,300:] = 0
            layer_edges = feature.canny(layer * 255)
            rows, cols = layer.shape
            
            """
            # Lines via Hough transform
            plt.figure()
            plt.imshow(layer_edges, cmap=plt.cm.gray)
            
            lines = transform.probabilistic_hough_line(layer_edges,
                                                       threshold=20,
                                                       line_length=10,
                                                       line_gap=50)            
            for line in lines:
                pt1, pt2 = line
                plt.plot((pt1[0], pt2[0]), (pt1[1], pt2[1]), '-r')
            plt.axis((0, cols, rows, 0))
            """
            
            rgb_values = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                          (128, 128, 128)]
            
            # Contours
            
            contours = measure.find_contours(layer, 0.5)
            #print('{} contours'.format(len(contours)))
            
            contours = [c[:,::-1] for c in contours]
            fn = '/Users/jonathan/block_coords/{}.csv'.format(frame_num)
            with open(fn, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
            
                for contour in contours:
                    
                    interior_pts = np.column_stack(draw.polygon(contour[:,1], contour[:,0]))
                    depth_pts = depth_frame[interior_pts[:,0], interior_pts[:,1]]
                    block_depth = (depth_offset - depth_pts).mean()
                    mean = interior_pts.mean(axis=0)
                    # I'm treating this like an average radius
                    std = interior_pts.std(axis=0)
                    if la.norm(std) < 5:
                        continue
                    interior_centered = interior_pts - np.vstack(interior_pts.shape[0] * (mean,))
                    U, S, V = la.svd(interior_centered)
                    D = np.zeros((U.shape[0], V.shape[0]))
                    for i in range(S.shape[0]):
                        D[i,i] = S[i]
                    rotated_contour = U.dot(D) + np.vstack(interior_pts.shape[0] * (mean,))
                    block_length = rotated_contour[:,0].max() - rotated_contour[:,0].min()
                    block_width = rotated_contour[:,1].max() - rotated_contour[:,1].min()
                    ratio = block_length / block_width
                    
                    cont_int = contour.astype(int)
                    colors = c_responses[interior_pts[:,0],interior_pts[:,1],:-1].sum(axis=0)
                    color = rgb_values[colors.argmax()]
                    
                    #axes[1].plot(contour[:, 0], contour[:, 1], color=color)
                    
                    pixel_to_metric = block_depth / f    # mm / pixel
                    if pixel_to_metric * block_length > 1.5 * block_length_metric:
                        continue
                    
                    if ratio < 1:
                        block_type = 'impossible'
                    elif ratio < 1.5:
                        # FIXME: Rotate corners by 45 degrees in this case
                        block_type = 'square'
                        diag_len = 0.5 * (block_length + block_width)
                        block_length = diag_len / 2 ** 0.5
                        block_width = diag_len / 2 ** 0.5
                    elif ratio < 3:
                        block_type = '2:1 rect'
                        block_width = block_length / 2
                    else:
                        block_type = '4:1 rect'
                        block_width = block_length / 4
                    
                    theta = np.rad2deg(np.arccos(V[0,0]))
                    row = list(pixel_to_metric * mean) + [block_depth, theta] \
                        + [block_width, block_length, block_height_metric] + list(color)
                    csvwriter.writerow(row)
                    
                    corners = np.array([[ block_length,  block_width],
                                        [ block_length, -block_width],
                                        [-block_length, -block_width],
                                        [-block_length,  block_width]]) / 2
                    corners = corners.dot(V) + np.vstack(corners.shape[0] * (mean,))
                    corners = np.vstack((corners, corners[0:1,:]))
                    
                    #print('Length: {:.2f}  |  Width: {:.2f}  |  Ratio: {:.2f}  |  {}'.format(block_length, block_width, ratio, block_type))
                    
                    #print('  {} | {} | {}'.format(contour.mean(axis=0),
                    #    la.norm(std), S))
                    
                    #colors = colors[colors[:,0,] != 128, :]
                    #axes[0].plot(corners[:, 1], corners[:, 0], color=color)
                    
                    #plt.plot(cc, rr, linewidth=2, color=color)
                    #line = np.vstack((mean, mean + 10 * V[0,:]))
                    #plt.plot(line[:,1], line[:,0], color='r')
                    #line = np.vstack((mean, mean + 10 * V[1,:]))
                    #plt.plot(line[:,1], line[:,0], color='b')
                
            """
            # Convex hull
            hull = morphology.convex_hull_object(layer_edges)
            objects = c_img.copy()
            objects[np.logical_not(hull), :] = np.zeros(3)
            #plt.figure()
            #plt.imshow(objects, cmap=plt.cm.gray)
            
            layers.append(objects)
            """
        """
        layers.append(np.zeros(layers[-1].shape, dtype='uint8'))
        layer_frame = np.vstack((np.hstack(tuple(layers[0:2])),
                                 np.hstack(tuple(layers[2:4]))))
        """
        """
        for ax in axes:
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
        
        #plt.figure()
        #plt.imshow(layer_frame, cmap=plt.cm.gray); plt.show()
        _, fn = os.path.split(rgb_path)
        #io.imsave(os.path.join(c.paths['working'], fn), layer_frame)
        plt.savefig(os.path.join(c.paths['working'], fn), format='png')
        plt.close()
        """
        
    
    """
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
    """
    
    