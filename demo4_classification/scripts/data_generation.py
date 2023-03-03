# Before running the script. Connect the camera via the USB3.0 cable to a USB3.0 port on your computer.
# Download pyrealsense2: pip install pyrealsense2

# Imports
import os
import cv2
import numpy
import pyrealsense2 as realsense

# SET THE NUMBER OF SAMPLES PER CLASS
number_of_samples_per_class = 100

# Camera setup params
color_resolution = (1920, 1080)
depth_resolution = (1280, 720)
frames_per_second = 30
id = '821312060313' # To be found in Realsense Viewer

# Folders
path_class_nut = "data/nut"
path_class_small_scratch = "data/small_scratch"
path_class_large_scratch = "data/large_scratch"
path_class_ok = "data/no_defect"

# Create folders
if not os.path.exists(path_class_nut): os.makedirs(path_class_nut)
if not os.path.exists(path_class_small_scratch): os.makedirs(path_class_small_scratch)
if not os.path.exists(path_class_large_scratch): os.makedirs(path_class_large_scratch)
if not os.path.exists(path_class_ok): os.makedirs(path_class_ok)

# Connect camera
conn = realsense.pipeline()

# Config camera
conf = realsense.config()
conf.enable_device(id)
conf.enable_stream(realsense.stream.depth, depth_resolution[0], depth_resolution[1], realsense.format.z16, frames_per_second)
conf.enable_stream(realsense.stream.color, color_resolution[0], color_resolution[1], realsense.format.bgr8, frames_per_second)
		
# Start streaming
conn.start(conf)

# Align images
align = realsense.align(realsense.stream.color)

# Data generation loop
counter = 1
print("\nData collection started.")
print("\nPress space bar to save a frame from the camera (click on image window)")
for i in os.listdir("data/"): # For all classes
    print("\nStart image collection of samples with " + i)
    while counter <= number_of_samples_per_class: # For all samples

        # Define path
        path = "data/" + i + "/sample_" + str(counter) + ".jpg"

        # Wait for image
        frames = conn.wait_for_frames()

        # Align images
        aligned_frames = align.process(frames)

        # Retreive images
        color_frame = aligned_frames.get_color_frame()

        # Convert to arrays
        color = numpy.asanyarray(color_frame.get_data())

        # Show image
        cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
        cv2.imshow('Data generation', color)
        k  = cv2.waitKey(1)

        # If space bar is pressed
        if k%256  == 32:
            cv2.imwrite(path, color)
            print('\tImage saved to ' + path)
            counter += 1

    # Reset counter for next class
    counter = 1

    # Print
    print("Stop image collection of samples with " + i)

# Exit
print("\nData collection ended.\n")
exit(0)



