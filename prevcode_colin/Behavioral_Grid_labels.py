
from pylab import *
import numpy as np
from skimage.draw import circle
# import cv2

n_wide = 4
n_long = 4
n_deep = 3

grid_size = 50
y_rez, x_rez = (n_deep*n_long*grid_size+100, n_wide*grid_size+50)
img = np.zeros([y_rez, x_rez], np.uint8)

whole_offset = (100/4, 100/4)
board_size = n_long*grid_size
offset = (50+whole_offset[0], 50+whole_offset[1])

# Keep list of points in all circles. Index using grid_pts[z][y][x]
grid_pts = []
# Setup points
for d in range(n_deep):
    pts_col = []
    for l in range(n_long):
        pts_row = []
        for w in range(n_wide):
            c = circle(d*board_size+w*grid_size+grid_size/2+whole_offset[0], 
                        l*grid_size+grid_size/2+whole_offset[1], 20)
            img[c[0], c[1]] = 255
            pts_row += [np.array(c).T]
        pts_col += [pts_row]
    grid_pts += [pts_col]
# Add lines between points
for d in range(1, n_deep):
    img[d*board_size-5+whole_offset[0]:d*board_size+5+whole_offset[0], :] = 255

# imshow(img)

cv2.imshow("viewer", img)

# TODO
# On the right have images of each block type with each orientation.
# The user should click on the block(/orientation) and the click on its place on the grid
# To do this, create another image. Then concatenate later?
# Select using number, not by clicking?

color_names = 'red', 'green', 'blue'
colors = {c:cm.colors.cnames[c] for c in color_names}

n_blocks = 5
block_width = 50
block_spacing = 100
offset = 25
img_shapes = np.zeros([400,(n_blocks+1)*(block_width+offset), 3])

block_2x2 = np.zeros([2*block_width, 2*block_width])
block_4x2 = np.zeros([2*block_width, 2*block_width])
block_2x4 = np.zeros([2*block_width, 2*block_width])
block_2x2[:block_width, :block_width] = 1
block_4x2[:, :block_width] = 1
block_2x4[:block_width, :] = 1
# block_2x4 = np.ones([block_width, 2*block_width])
# block_4x2 = np.ones([2*block_width, block_width])

for i, c in enumerate(color_names):
    color = matplotlib.colors.hex2color(colors[c])
    h=0
    img_shapes[h*block_spacing+(h+1)*offset:(h+1)*block_spacing+(h+1)*offset,\
                i*block_spacing+(i+1)*offset:(i+1)*block_spacing+(i+1)*offset] = \
                block_2x2[:,:,None] * color
    h=1
    img_shapes[h*block_spacing+(h+1)*offset:(h+1)*block_spacing+(h+1)*offset,\
                i*block_spacing+(i+1)*offset:(i+1)*block_spacing+(i+1)*offset] = \
                block_2x4[:,:,None] * color
    h=2
    img_shapes[h*block_spacing+(h+1)*offset:(h+1)*block_spacing+(h+1)*offset,\
                i*block_spacing+(i+1)*offset:(i+1)*block_spacing+(i+1)*offset] = \
                block_4x2[:,:,None] * color

cv2.imshow("shapes", img_shapes)
cv2.waitKey(30)

import matplotlib.colors
matplotlib.colors.hex2color




# cv2.imshow("grid", img)
# cv2.waitKey(30)


