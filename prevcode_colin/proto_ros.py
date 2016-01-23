#!/usr/bin/python

"""
"""

import numpy as np
import optparse
import rospy
import tf

import cv2
from pyKinectTools.utils.pointcloud_conversions import *
# from pyKinectTools.utils.transformations import *

from sensor_msgs.msg import PointCloud2
from predicator_msgs.msg import *
from std_msgs.msg import Empty
from predicator_8020_module.utils_8020 import *

from threading import Lock
from copy import deepcopy

# Images
im_display = None
im_pos = None
im_depth = None
im_rgb = None
mask = np.ones([480, 640], bool)
image_lock = Lock()

# 8020 Detector
# display = False
# clf_mean = None
# clf_w = None
# closest_parts = {x:None for x in [3,4,8]}
# pub_list = None


def draw_box(img, box):
    """ Function to draw the rectangle """
    cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0))


def mouse_event(event, x, y, flags=None, params=None):
    # global bounding_box
    if event == cv2.EVENT_LBUTTONDOWN:
        # bounding_box += [[y, x]]
        params += [[y, x]]

    return params


def pick_block(im, block_id=None):
    points = []
    cv2.setMouseCallback("pick_region", mouse_event, points)
    cv2.namedWindow("pick_region")

    im_display = im.copy()
    while len(points) < 1:
        # Display instructions
        txt = "Click on block"
        if block_id is not None:
            txt += " " + block_id

        cv2.putText(im_display, txt, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

        cv2.imshow("pick_region", im_display)
        ret = cv2.waitKey(30)
    cv2.destroyWindow("pick_region")

    return points[0]


def pointcloud_callback(data):
    rospy.logwarn('got image')
    """
    data : ros msg data
    """
    # global im_display
    global im_pos, im_depth, im_rgb, image_lock

    with image_lock:
        frame_time = data.header.stamp
        frame_seq = data.header.seq

        x = pointcloud2_to_array(data, split_rgb=True)
        im_pos = np.dstack([x['x'], x['y'], x['z']])
        im_pos = np.nan_to_num(im_pos)
        im_depth = im_pos[:, :, 2]

        im_rgb = np.dstack([x['r'], x['g'], x['b']])
        # im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)


# def process_plate_detector(data):
#     global im_display
#     global im_pos, im_depth, im_rgb, bounding_box, mask
#     global frame_time, frame_seq
#     global clf_mean, clf_w
#     global closest_parts
#     global pub_list

#     print "Trying to process plate"
#     if im_rgb is None:
#         return False
#     print "Processing plate"

#     try:
#         with image_lock:
#             im, mask = extract_foreground_poly(im_rgb, bounding_box)
#             # clf_mean, clf_w = train_clf(im, mask)
#             pred_mask, objects, props = get_foreground(im, clf_mean, clf_w)
#             all_centroids, all_holes = extract_holes(im, pred_mask, objects, props, im_pos, im_rgb)

#             print "# Plates:", len(all_holes)
#             if len(all_centroids) > 0:
#                 closest_parts, classes = get_closest_part(all_centroids, all_holes)
#                 im_display = plot_holes(im_rgb, all_holes)
#                 # im_display = (pred_mask > 0)*255

#             else:
#                 for c in closest_parts:
#                     closest_parts[c] = None

#             ps = PredicateList()
#             ps.pheader.source = rospy.get_name()
#             ps.statements = []

#             # Send predicates for each plate
#             for i in [3,4,8]:
#                 # Check if the plate is available
#                 if closest_parts[c] is None:
#                     continue
#                 # Setup/send predicate
#                 plate_name = "plate_{}".format(i)
#                 statement = PredicateStatement(predicate=plate_name,
#                                                     confidence=1,
#                                                     value=PredicateStatement.TRUE,
#                                                     num_params=1,
#                                                     params=[predicate_param, "", ""])
#                 ps.statements += [statement]
#             pub_list.publish(ps)


#         return True
#     except:
#         return False


if __name__ == '__main__':

    try:
        parser = optparse.OptionParser()
        parser.add_option("-c", "--camera", dest="camera",
                          help="name of camera", default="camera")
        parser.add_option("-n", "--namespace", dest="namespace",
                          help="namespace for occupancy data", default="")    
        (options, args) = parser.parse_args()

        camera_name = options.camera
        namespace = options.namespace
    except:
        camera_name = 'camera'

    # Setup ros/publishers
    rospy.init_node('block_module')

    # Setup subscribers
    cloud_uri = "/{}/depth_registered/points".format(camera_name)
    rospy.Subscriber(cloud_uri, PointCloud2, pointcloud_callback, queue_size=10)

    # Get occupancy params
    # occupancy_center = np.array(rospy.get_param("/{}/occupancy_center".format(namespace)))
    # occupancy_radius = np.array(rospy.get_param("/{}/occupancy_radius".format(namespace)))

    # Setup TF
    # tf_broadcast = tf.TransformBroadcaster()


    rate = rospy.Rate(30)
    rate.sleep()

    # while im_rgb is None or im_pos is None:
    #     rate.sleep()


    display = True
    print "Ready"
    # ret = process_plate_detector([])

    while not rospy.is_shutdown():
        print "Running1"
        while im_pos == None:
            rate.sleep()

        ret = process_plate_detector([])
        # Show colored image to reflect if a space is occupied
        if display and im_display is not None:
            print "Display"
            cv2.imshow("img", im_display)
            cv2.waitKey(30)
        # else:
            # print "No display image"

        # Send TFs for each plate at every timestep
        for c in closest_parts:
            plate_name = "plate_{}".format(c)
            if closest_parts[c] is None:
                x, y, z = [-1, -1, -1]
            else:
                x, y, z = closest_parts[c]

            tf_broadcast.sendTransform([x, y, z],
                                        [0,0,0,1],
                                        rospy.Time.now(), plate_name, 
                                        "camera_rgb_optical_frame")

        rate.sleep()
