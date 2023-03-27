# Before running the script. Connect the camera via the USB3.0 cable to a USB3.0 port on your computer.
# Download pyrealsense2: pip install pyrealsense2

# Imports
import os
import cv2
import numpy
import pyrealsense2 as realsense

# Camera setup params
color_resolution = (1920, 1080)
depth_resolution = (1280, 720)
frames_per_second = 30
camera_id = '821312060313'  # To be found in Realsense Viewer

# Root folder
root = "data/defect_images"

# Folders
path_class_nut = f"{root}/nut_new"
path_class_large_scratch = f"{root}/scratch_new"
path_class_ok = f"{root}/no_defect_new"

# Create folders
if not os.path.exists(path_class_nut):
    os.makedirs(path_class_nut)
if not os.path.exists(path_class_large_scratch):
    os.makedirs(path_class_large_scratch)
if not os.path.exists(path_class_ok):
    os.makedirs(path_class_ok)

# Connect camera
conn = realsense.pipeline()

# Config camera
conf = realsense.config()
conf.enable_device(camera_id)
conf.enable_stream(realsense.stream.depth, depth_resolution[0], depth_resolution[1], realsense.format.z16,
                   frames_per_second)
conf.enable_stream(realsense.stream.color, color_resolution[0], color_resolution[1], realsense.format.bgr8,
                   frames_per_second)

# Start streaming
conn.start(conf)

# Align images
align = realsense.align(realsense.stream.color)

# Data generation loop
print("Data collection started.\n")
print("Press space bar to save a frame from the camera (click on image window)\n")
for i in os.listdir(f"{root}/"):  # For all classes
    print("Start image collection of samples with " + i)
    if not os.listdir(f"{root}/" + i):
        counter = 0
        print(f"Currently no images in this class")
    else:
        counter = os.listdir(f"{root}/" + i).__len__()
        print(f"Currently {counter} images in this class")
    while True:  # Until ESC is pressed

        # Define path
        path = f"{root}/{i}/sample_" + str(counter+1) + ".jpg"

        # Wait for image
        frames = conn.wait_for_frames()

        # Align images
        aligned_frames = align.process(frames)

        # Retrieve images
        color_frame = aligned_frames.get_color_frame()

        # Convert to arrays
        color = numpy.asanyarray(color_frame.get_data())

        # Show image
        cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
        cv2.imshow('Data generation', color)
        k = cv2.waitKey(1)

        # If space bar is pressed
        if k % 256 == 32:
            cv2.imwrite(path, color)
            print('\tImage saved to ' + path)
            counter += 1
        elif k % 256 == 27:  # ESC to go to next class or exit
            break

    # Print
    print(f"Stop image collection of samples with {i}\n")

# Exit
print("Data collection ended.")
exit(0)
