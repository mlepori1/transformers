# Convert rosbag w/ pointclouds from Kinect into a set of pngs

import os
import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge
from pyKinectTools.utils.pointcloud_conversions import pointcloud2_to_array

folder_dir = os.path.expanduser("~/Desktop/Blocks/Oct28/")
files = os.listdir(folder_dir)
files = filter(lambda x:".bag" in x, files)

for fid in files:
    trial_name = fid[:-4]

# if 1:
    # trial_name = files[-1][:-4]    
    # uri_in = os.path.expanduser("~/Desktop/Blocks/Block_test_good"+".bag")
    # uri_out = os.path.expanduser("~/Desktop/Blocks/tmp/")
    # uri_in = os.path.expanduser("~/Desktop/Blocks/"+".bag")
    uri_in = folder_dir + trial_name + ".bag"
    uri_out = folder_dir + "/frames/" + trial_name + "/"
    bag = rosbag.Bag(uri_in)

    if not os.path.isdir(uri_out):
        os.mkdir(uri_out)

    # Get topic names
    if 0:
        gen = bag.read_messages()

        topics = []
        for i in range(100):
            topics += [gen.next()[0]]
        topics = np.unique(topics)

    # Using a Pointcloud2 object
    if 0:
        gen_pts = bag.read_messages("/camera/depth_registered/points")

        while 1:
            try:
                _, data, t = gen_pts.next()
            except:
                print "Error"
                break

            x = pointcloud2_to_array(data, split_rgb=True)
            im_pos = np.dstack([x['x'], x['y'], x['z']])
            im_rgb = np.dstack([x['r'], x['g'], x['b']])[:,:,[2,1,0]]

            fid_time = str(t.to_time())
            fid_rgb = "{}/{}_rgb.png".format(uri_out, fid_time)
            fid_xyz = "{}/{}_xyz.npz".format(uri_out, fid_time)

            # Save images as PNG (rgb) and NumPy (xyz) file
            # -- Uncompressed saves 32.3 in ms
            # -- Compressed saves in 120 ms
            cv2.imwrite(fid_rgb, im_rgb)
            np.savez(fid_xyz, im_pos)
            np.savez_compressed(fid_xyz, im_pos)

            # Verify the data
            # -- Uncompressed loads in 3.13 ms        
            # -- Compressed loads in 14.4 ms
            # xyz = np.load(fid_xyz)['arr_0']
            
    bridge  = CvBridge()
    if 1:
        # Using a Pointcloud2 object
        # gen_xyz = bag.read_messages("camera/depth_registered/image_raw")
        gen_rgb = bag.read_messages("camera/rgb/image_raw")
        
        for a, data_rgb, t in gen_rgb:
        # while 1:            
            # try:
                # a, data_rgb, t = gen_rgb.next()
                # _, data_xyz, t = gen_xyz.next()
            # except:
                # print "Error"
                # break

            im_rgb = bridge.imgmsg_to_cv2(data_rgb, "bgr8")


            # im_rgb = np.zeros([480,640,3], np.uint8)
            # im_rgb = np.fromstring(data_rgb.data, np.uint16).reshape([480, 640])
            # im_rgb[:,:,:2] = np.fromstring(data_rgb.data, np.uint8).reshape([480, 640, 2])
            # im_rgb = im_rgb[:,:,None]
            # im_rgb_out = np.zeros([480,640,3], np.uint8)
            # im_rgb_out = cv2.cvtColor(im_rgb, cv2.cv.CV_YCrCb2BGR)
            # im_rgb_out = cv2.cvtColor(im_rgb, cv2.cv.CV_BayerGB2RGB)
            # im_rgb_out = cv2.cvtColor(im_rgb, cv2.cv.CV_BayerRG2RGB)
            # im_rgb_out = cv2.cvtColor(im_rgb, cv2.cv.CV_BayerRG2RGB)
            # im_xyz = np.frombuffer(data_xyz.data, np.float32).reshape([480,320])

            # x = pointcloud2_to_array(data, split_rgb=True)
            # im_pos = np.dstack([x['x'], x['y'], x['z']])
            # im_rgb = np.dstack([x['r'], x['g'], x['b']])[:,:,[2,1,0]]

            fid_time = str(t.to_time())
            fid_rgb = "{}/{}_rgb.png".format(uri_out, fid_time)
            # fid_xyz = "{}/{}_xyz.npz".format(uri_out, fid_time)

            # Save images as PNG (rgb) and NumPy (xyz) file
            # -- Uncompressed saves 32.3 in ms
            # -- Compressed saves in 120 ms
            cv2.imwrite(fid_rgb, im_rgb)
            # np.savez(fid_xyz, im_pos)
            # np.savez_compressed(fid_xyz, im_pos)

            # Verify the data
            # -- Uncompressed loads in 3.13 ms        
            # -- Compressed loads in 14.4 ms
            # xyz = np.load(fid_xyz)['arr_0']
            



