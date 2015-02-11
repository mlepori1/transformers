# blocks

## DATA COLLECTION
roslaunch openni_launch openni.launch depth_registration:=true
rosbag record camera/depth_registered/image_raw camera/depth_registered/camera_info camera/rgb/image_raw camera/rgb/camera_info -o kinect
roslaunch openni_launch openni.launch load_driver:=false
