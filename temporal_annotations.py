
import os
import time
import numpy as np
import cv2

# Note: Save to text file, not npz

folder =  os.path.expanduser("~/Desktop/Blocks/Oct28/frames/")

video_fids = os.listdir(folder)
video_fids = filter(lambda x:".DS_" not in x, video_fids)

# Go through each folder and check if annotations have already been created.
vid = 0
annotations = [1]
while vid < len(video_fids) and len(annotations) >= 1:
    full_folder = folder+video_fids[vid]
    image_fids = os.listdir(full_folder)
    annotations = filter(lambda x:"temporal_segmentation_" in x, image_fids)
    vid += 1
image_fids = filter(lambda x:".DS_" not in x, image_fids)
image_fids = filter(lambda x:"temporal_segmentation_" not in x, image_fids)

# image_fids = filter(lambda x:".DS_" not in x, image_fids)
n_images = len(image_fids)

# Setup image window
cv2.namedWindow("Segmentation")


timeline = np.zeros(n_images)
markers = []
from scipy.signal import resample

frame = 0
while frame >= 0 and frame < n_images:
    image_name = image_fids[frame]
    img = cv2.imread(full_folder+"/"+image_name)#[:,:,[2,1,0]]

    # img[-100:,timeline[sample_pts]>0] = 255
    t = resample(timeline, img.shape[1])>0.5
    img[-100:,t] = 255
    if image_name in markers:
        sample_pts = np.linspace(0, n_images, img.shape[1]).astype(np.int)    
        img[-50:,:] = [128, 0,0]


        # img[-50:,:] = [128, 0,0]

    # TXT: Add name of file
    img_show = img.copy()

    # Add indicator for marker at current frame

    # TXT:  Add instructions
    txt = "Back: <-, Forward: ->"
    cv2.putText(img_show, txt, (5, 25), cv2.FONT_HERSHEY_DUPLEX, 1., (0, 0, 0))
    txt = "Space: start/end of action"
    cv2.putText(img_show, txt, (5, 50), cv2.FONT_HERSHEY_DUPLEX, 1., (0, 128, 128))
    txt = "q: quit"
    cv2.putText(img_show, txt, (5, 75), cv2.FONT_HERSHEY_DUPLEX, 1., (0, 0, 128))

    # Display image
    ret = -1
    escape_chars = ["\r", "\x1b", " ", 'q']
    escape_chars = [ord(x) for x in escape_chars]
    arrows = [63234, 63235] # left/right
    while ret == -1 or (not (ret in arrows) and not (ret in escape_chars)):
        cv2.imshow("Segmentation", img_show)
        ret = cv2.waitKey(100)

    if ret == ord(" "):
        if image_name in markers:
            markers.remove(image_name)
            timeline[frame] = 0
        else:
            markers.append(image_name)
            timeline[frame] = 1
    elif ret == 63234: 
        #left arrow
        frame -= 1
    elif ret == 63235: 
        #left arrow
        frame += 1 
    elif ret == ord('q'): 
        break       

with open(full_folder+"/temporal_segmentation_{}.txt".format(time.time()), 'w') as fid:
    for m in markers:
        fid.write("{}\n".format(m))
# np.savetxt(full_folder+"temporal_segmentation.txt", markers)
txt = "END OF SEQUENCE"
cv2.putText(img_show, txt, (100, 240), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255))
cv2.imshow("Segmentation", img_show)
cv2.waitKey(10)
print "End of sequence"


