"""
labelvideos.py
  Interactive GUI for assigning temporal annotations using the RGB video from
  a trial

AUTHOR
  Jonathan D. Jones
"""

import os
import glob
import numpy as np
import cv2


def labelVideo(trial_id):
    """
    [DESCRIPTION]
    
    Args:
    -----
    [str] trial_id:
    
    Returns:
    --------
      (Nothing)
    """
    
    rgb_path = os.path.join('data', 'rgb', trial_id)
        
    # Create label directory if it doesn't exist
    label_path = os.path.join('data', 'rgb', trial_id)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    # Get full path names for RGB frames in this trial
    pattern = os.path.join(rgb_path, '*.png')
    rgb_frame_fns = glob.glob(pattern)
    num_frames = len(rgb_frame_fns)
    
    # Set up image window
    cv2.namedWindow('Segmentation')
    
    frame_idx = 0
    while frame_idx >= 0 and frame_idx < num_frames:
        
        # Open the current frame and display it
        frame_path = rgb_frame_fns[frame_idx]
        frame = cv2.imread(frame_path)
        cv2.imshow('Segmentation', frame)
            
        # Wait until we read some input from the user
        ret = cv2.waitKey(0)
            
        # Act on the input if it's one of the recognized characters
        if ret == ord(" "):
            print('space')
        elif ret == ord('j'):   # Jump five frames ahead
            frame_idx -= 5
        elif ret == ord('k'):   # Jump five frames back
            frame_idx += 5
        elif ret == ord('q'):   # Exit
            break       
    
    print("End of sequence")


if __name__ == '__main__':
    trial_id = '1460754570'
    labelVideo(trial_id)


