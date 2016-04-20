

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import closing, convex_hull_image
from skimage.color import *
import time
from geometry_msgs.msg import Point32, TransformStamped





base_dir = "/Users/colin/Desktop/Blocks/"

def load_model(filename):
    """ From xyz file """
    model_pts = np.fromfile(filename, np.float, sep=" ").reshape([-1, 3])/100.
    model_pts -= model_pts.mean(0)
    model_pts *= [1,1,-1]
    return model_pts

model_names = {
    "2x2": "/Users/colin/Desktop/Blocks/CAD_pts/Lego_2x2_hollow.xyz",
    "4x2": "/Users/colin/Desktop/Blocks/CAD_pts/Lego_4x2_hollow.xyz"
}

# blue_color = np.array([ 8, 42, 105], dtype=np.uint8)
# green_color = np.array([ 14, 103,  47], dtype=np.uint8)
# green_color = np.array([ 50, 120,  30], dtype=np.uint8)

class Block:
    cad_name = None
    color = None
    
    # length = 1
    # width = 1

    xyz_hist = None
    xy = None
    xyz = None
    

    def __init__(self, color, cad_name=None):
        self.color = color
        if cad_name is not None:
            self.cad_name = cad_name
        
        self.xyz_hist = []
        self.xy = np.array([-1, -1], np.float)
        self.xyz = np.array([-1, -1, -1], np.float)


    def update_position(self, xyz):
        self.xyz = xyz
        self.xyz_hist += [xyz]

    def reset(self):
        self.xyz_hist = []


class BlockSet:
    colors = {}
    colors_std = {}
    cad_models = {}
    blocks = []

    def __init__(self):
        pass

    def add_color(self, name, color, std):
        self.colors[name] = color
        self.colors_std[name] = std

    def add_block(self, block_new):
        self.blocks += [block_new]

    def n_blocks(self, color=None, return_idx=False):
        """ Returns the number of blocks (optionally of a specifed color) """
        if color is None:
            # Count for any color
            n_blocks = len(self.blocks)
            if not return_idx:
                return n_blocks
            else:
                return n_blocks, range(n_blocks)
        else:
            # Find only the selected color
            n_all_blocks = len(self.blocks)
            colored_block_idxs = [i for i,x in zip(range(n_all_blocks), self.blocks) if x.color==color]
            n_blocks = len(colored_block_idxs)

            if not return_idx:
                return n_blocks
            else:
                return n_blocks, colored_block_idxs


    def get_block_poses(self, color=None, return_idx=False):
        """ Returns the block positions (optionally of a specifed color) """
        if color is None:
            # Count for any color
            n_blocks = len(self.blocks)
            poses = [x.xyz for x in self.blocks]
            if not return_idx:
                return poses
            else:
                return poses, range(n_blocks)
        else:
            # Find only the selected color
            n_all_blocks = len(self.blocks)
            colored_block_idxs = [i for i,x in zip(range(n_all_blocks), self.blocks) if x.color==color]
            poses = [x.xyz for x in self.blocks if x.color==color]

            if not return_idx:
                return poses
            else:
                return poses, colored_block_idxs                

    def reset_blocks(self):
        for b in self.blocks:
            b.reset()

    def update_position(self, idx, xyz):
        self.blocks[idx].update_position(xyz)

    def add_cad(self, name, filename):
        self.cad_models[name] = load_model(filename)


import sklearn.utils.linear_assignment_ as sk_assignment
from sklearn.metrics.pairwise import pairwise_distances

def object_association(new_objects, prev_objects):
    """
    Find matches between new objects and previous objects
    --Input--
    new_objects : list of RGBDObject
    """
    # Get number of objects
    n_new_obj = len(new_objects)
    n_total_obj = len(prev_objects)
    MAX = 999999.

    # Get positions of previous objects (pad if #new > #old)
    prev_obj_pos = np.zeros([n_total_obj, 3], np.float) + MAX
    for i in xrange(n_total_obj):
        prev_obj_pos[i] = prev_objects[i]
        # objects[i].active = False

    # if we don't have any object detections, no need to continue
    # if n_new_obj == 0:
        # return

    # new_obj_pos = np.vstack([x.pos for x in new_objects])
    new_obj_pos = new_objects

    # Get assignment of new objects to previous objects
    dists = pairwise_distances(new_obj_pos, prev_obj_pos)
    assignment = sk_assignment.linear_assignment(dists)

    return assignment

    # # Update previous object positions and add any new objects
    # for i, j in assignment:
    #     if j < self.n_objects:
    #         self.objects[j].active = True
    #         self.objects[j].pos = new_objects[i].pos
    #         self.objects[j].ori = new_objects[i].ori
    #         self.objects[j].seq = new_objects[i].seq
    #     else:
    #         self.objects += [new_objects[i]]

    # # Remove old objects
    # # self.objects = [obj for obj in self.objects if obj.seq >= self.frame_seq-self.MAX_PREV_SEQ]

    # for i in range(len(self.objects)):
    #     # self.objects[i].seq = True if self.objects[i].seq == self.frame_seq else False
    #     self.objects[i].seq = True

    # self.n_objects = len(self.objects)



# Initialize blocks
block_set = BlockSet()

# Load CAD files
for name,filename in model_names.items():
    block_set.add_cad(name, filename)

# Setup blocks in image
block_set.add_block(Block("green", cad_name='2x2'))
block_set.add_block(Block("green", cad_name='2x2'))
block_set.add_block(Block("blue", cad_name='4x2'))
block_set.add_block(Block("blue", cad_name='4x2'))

############## SCRIPT #################
if 0:
    im = cv2.imread(base_dir + "Blocks.jpg")[:,:,[2,1,0]]
    im = cv2.resize(im, (640, 480))
else:
    # Wait for ros msg
    im = im_rgb[:,:,[2,1,0]]
    im = im_rgb

if 0:
    # Get background model
    n = 1
    ims = np.zeros([480,640, 3, n])
    for i in range(n):
        ims[:,:,:,i] = im_rgb[:,:,[2,1,0]]
        # ims += [im_new]
        time.sleep(.1)
    im_bg = ims.mean(-1).astype(np.uint8)

# Pick blocks to get colors
if 0:
    # Setup colors
    green_loc = pick_block(im, "#1")
    time.sleep(.5)
    blue_loc = pick_block(im, "#2")

    im_ = rgb2lab(im)
    im_[:,:,0] *= 0#im_[:,:,1]
    # im_fg = im_[:,:,1]

    green_color = im_[green_loc[0], green_loc[1]]
    blue_color = im_[blue_loc[0], blue_loc[1]]


color_variance = 10
block_set.add_color("green", green_color, color_variance)
block_set.add_color("blue", blue_color, color_variance)

block_set.reset_blocks()
T=1000
for t in range(T):
    with image_lock:
        # im = im_rgb.copy()[:,:,[2,1,0]]
        im = im_rgb.copy()
        im_ = rgb2lab(im)
        im_[:,:,0] *= 0#im_[:,:,1]
        im_pos_t = im_pos.copy()


    im_out = im.copy()
    # Compute responses based on color
    for b in block_set.colors:
        response = np.linalg.norm(im_*1. - block_set.colors[b], 2, axis=-1) / float(block_set.colors_std[b])
        # block_responses[b]
        valid = response < 3
        # valid = closing(valid, np.ones([3,3]))

        labels = label(valid)*valid
        label_info = regionprops(labels)

        # Make sure the components are big enough
        label_info = filter(lambda x:500 < x.area < 10000, label_info)

        # Get the top n blocks
        n_blocks, block_idxs = block_set.n_blocks(b, return_idx=True)
        sorted_blocks_idxs = np.argsort([x.area for x in label_info])
        label_info = [label_info[x] for x in sorted_blocks_idxs]
        n_new_blocks = len(label_info)

        if n_new_blocks == 0:
            continue

        new_obj_poses = []
        new_obj_ids = []
        if t > 0:
            prev_obj_poses, prev_obj_ids = block_set.get_block_poses(b, return_idx=True)

            # Match previous objects to new objects
            for i,l in enumerate(label_info):
                xyz = im_pos_t[int(l.centroid[0]), int(l.centroid[1])]
                new_obj_poses += [xyz]
                new_obj_ids += [block_idxs[i]]

            assoc = object_association(new_obj_poses, prev_obj_poses)
            for i,j in assoc:
                block_set.blocks[prev_obj_ids[j]].update_position(new_obj_poses[i])
                # block_set.blocks[prev_obj_ids[i]].update_position(new_obj_poses[j])

                # Color the block
                id_normed = prev_obj_ids[j] / float(len(block_set.blocks))
                color = ( np.array(cm.jet(id_normed)[:3]) *255 ).astype(np.uint8)
                coords = label_info[i].coords
                im_out[coords[:,0], coords[:,1]] = color

                # Register model
                idx = i                
                coords = label_info[idx].coords
                pts = im_pos_t[coords[:,0], coords[:,1]]
                pts = pts[np.all(pts!=0, -1)]
            
                block_idx = block_idxs[idx]
                cad_name = block_set.blocks[block_idx].cad_name
                model_pts = block_set.cad_models[cad_name].copy()
                register_model(pts, model_pts, color=color)



        else:
            for i,l in enumerate(label_info):
                block_idx = block_idxs[i]
                center = l.centroid
                block_set.blocks[block_idx].xy = center
                xyz = im_pos[int(center[0]), int(center[1])]
                new_obj_poses += [xyz]
                
                # block_set.blocks[block_idx].xyx = xyz
                # block_set.blocks[block_idx].save_position()
                block_set.blocks[block_idx].update_position(xyz)
                # block_set.update_position(block_idx, xyz)
                
                id_normed = block_idx / float(len(block_set.blocks))
                color = ( np.array(cm.jet(id_normed)[:3]) *255 ).astype(np.uint8)
                # color = block_set.colors[b]
                im_out[l.coords[:,0], l.coords[:,1]] = color#50*(block_idxs[i]+1)
                # print b, block_idx, xyz


    cv2.imshow("im_out", im_out[:,:,[2,1,0]])
    ret = cv2.waitKey(30)
    if ret >= 0:
        break

# figure(0)
# imshow(im_out)

if 0:
    #  Plot points in block_set
    figure(1)
    for b in block_set.blocks:
        print b.xy
        scatter(b.xy[1], -b.xy[0], s=35, color=b.color[0])

    axis([0,640,0,-480])

if 0:
    figure(1)
    colors = 'rgbk'
    for i,b in enumerate(block_set.blocks):
        trail = np.array(b.xyz_hist)
        plot(trail[:,0], trail[:,1], c=colors[i])
        # scatter3d(trail)
        print i, trail.shape, trail[-1]

if 0:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # figure(1)
    colors = 'rgbk'
    for i,b in enumerate(block_set.blocks):
        trail = np.array(b.xyz_hist)
        # plot(trail[:,0], trail[:,1], c=colors[i])
        valid = np.nonzero(np.all(trail != 0, -1))[0]
        ax.plot3D(trail[valid,0], trail[valid,1], trail[valid,2], c=colors[i])
        print i, trail.shape, trail[-1]
        

# Export pts
if 0:
    all_pts = (im_pos_t*valid[:,:,None]).reshape([-1, 3])
    all_pts = all_pts[np.all(all_pts!=0, -1)]
    with open("/Users/colin/Desktop/tmp_pts.xyz", 'w') as fid:
        for pt in all_pts:
            fid.write("{} {} {}\n".format(*pt))
            # fid.write("{} {} {} {} {} {}\n".format(*pt))


if 0:
    pub_blocks = rospy.Publisher("blocks", PointCloud2)
    pub_scene = rospy.Publisher("scene", PointCloud2)
    tf_man = tf.Transformer(True, rospy.Duration(1000.0))

    m = TransformStamped()
    m.header.frame_id = "camera_rgb_optical_frame"
    m.child_frame_id = "blocks"
    m.transform.translation.x = 0
    m.transform.translation.y = 0
    m.transform.translation.z = 0
    m.transform.rotation.x = 0
    m.transform.rotation.y = 0
    m.transform.rotation.z = 0
    m.transform.rotation.w = 1
    tf_man.setTransform(m)

# Load lego model
if 0:
    # filename = "/Users/colin/Desktop/Blocks/CAD_pts/Lego_2x2_hollow.xyz"
    # filename = "/Users/colin/Desktop/Blocks/CAD_pts/Lego_4x2_hollow.xyz"
    # model_pts = np.fromfile(filename, np.float, sep=" ").reshape([-1, 3])/100.
    # model_pts -= model_pts.mean(0)
    # model_pts *= [1,1,-1]

    
    idx = 0
    block_idx = block_idxs[idx]
    cad_name = block_set.blocks[block_idx].cad_name
    model_pts = block_set.cad_models[cad_name].copy()
    # model_pts *= [1,1,-1]
    # model_pts *= [-1,1,-1]
    
    coords = label_info[idx].coords
    pts = im_pos_t[coords[:,0], coords[:,1]]
    pts = pts[np.all(pts!=0, -1)]
        

    def filter_cloud(pts, norm_dist=.1):
        """ Zero-center the pointcloud, compute the norm, and remove outliers """
        # Center the data
        pts_mean = pts.mean(0)
        pts -= pts_mean
        
        # Remove outliers
        norms = np.linalg.norm(pts, axis=1)
        pts = pts[norms < .1]
        
        # Put back in original frame
        pts += pts_mean
        
        return pts

    pts = filter_cloud(pts, 0.1)

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(pts[::1,0], pts[::1,1], pts[::1,2])


    # from pyKinectTools.algs.IterativeClosestPoint import IterativeClosestPoint
    pts_mean = pts.mean(0)
    pts -= pts_mean

    # Match model to points
    if 1:
        R, t = IterativeClosestPoint(model_pts[::10], pts[::10], min_change=0, pt_tolerance=.1, max_iters=20)
        # R, t, pts_m = IterativeClosestPoint(model_pts[::2], pts, pt_tolerance=1, max_iters=50, return_transform=True)
        # pts_t = np.dot(R, pts.T).T + t
        pts_m = np.dot(R, model_pts.T).T + t
    else:
        # Match points to model
        R, t = IterativeClosestPoint(pts, model_pts[::1], pt_tolerance=.5, max_iters=50)
        # pts_t = np.dot(R.T, (pts-t).T).T    
        pts_m = np.dot(R.T, (model_pts-t).T).T

    print t
    print R

    i = 0

    pts_model_scene = pts_m + pts_mean
    # pts_model_scene = pts_m 
    dtype = np.dtype([('x',np.float32), ('y',np.float32), ('z',np.float32)])
    pts_model_scene_pc2 = np.empty(pts_model_scene.shape[0], dtype=dtype)
    pts_model_scene_pc2['x'] = pts_model_scene[:,0]
    pts_model_scene_pc2['y'] = pts_model_scene[:,1]
    pts_model_scene_pc2['z'] = pts_model_scene[:,2]

    pt_model_scene_pc2_ = array_to_pointcloud2(pts_model_scene_pc2)
    # pt_model_scene_pc2_.header.frame_id = "blocks"
    pt_model_scene_pc2_.header.frame_id = "camera_rgb_optical_frame"
    pt_model_scene_pc2_.header.stamp = rospy.Time.now()
    pt_model_scene_pc2_.header.seq = i
    i += 1

    pub_blocks.publish(pt_model_scene_pc2_)
    pointcloud2_to_array(pt_model_scene_pc2_)

    # pts_t += pts_mean
    from matplotlib.pylab import *
    from mpl_toolkits.mplot3d import axes3d
    ax = plt.axes(projection='3d')
    ax.scatter3D(pts[::1,0], pts[::1,1], pts[::1,2], c='b')
    ax.scatter3D(pts_m[::1,0], pts_m[::1,1], pts_m[::1,2], c='g')
    # ax.scatter3D(model_pts[::1,0], model_pts[::1,1], model_pts[::1,2], c='k')
    axis('equal')


def register_model(pts, model_pts, color=None):
    global pub_blocks

    pts_mean = pts.mean(0)
    pts -= pts_mean

    # Match model to points
    R, t = IterativeClosestPoint(model_pts[::10], pts[::10], min_change=0, pt_tolerance=.1, max_iters=20)
    pts_m = np.dot(R, model_pts.T).T + t

    pts_model_scene = pts_m + pts_mean
    if color is None:
        dtype = np.dtype([('x',np.float32), ('y',np.float32), ('z',np.float32)])
    else:
        dtype = np.dtype([('x',np.float32), ('y',np.float32), ('z',np.float32),
                          ('r',np.uint8), ('g',np.uint8), ('b',np.uint8)])
    pts_model_scene_pc2 = np.empty(pts_model_scene.shape[0], dtype=dtype)

    pts_model_scene_pc2['x'] = pts_model_scene[:,0]
    pts_model_scene_pc2['y'] = pts_model_scene[:,1]
    pts_model_scene_pc2['z'] = pts_model_scene[:,2]
    
    if color is None:
        pt_model_scene_pc2_ = array_to_pointcloud2(pts_model_scene_pc2)
    else:
        pts_model_scene_pc2['r'] = color[0]
        pts_model_scene_pc2['g'] = color[1]
        pts_model_scene_pc2['b'] = color[2]
        pt_model_scene_pc2_ = array_to_pointcloud2(pts_model_scene_pc2, merge_rgb=True)
    
    pt_model_scene_pc2_.header.frame_id = "camera_rgb_optical_frame"
    pt_model_scene_pc2_.header.stamp = rospy.Time.now()

    pub_blocks.publish(pt_model_scene_pc2_)

    return



