# Imports
import cv2
import time
import numpy
import config
import pyrealsense2 as realsense

# Connect camera
conn = realsense.pipeline()

# Camera setup params
color_resolution = config.color_resolution
depth_resolution = config.depth_resolution
frames_per_second = config.frames_per_second
id = config.id

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
while True:

    # Wait for image
    frames = conn.wait_for_frames()

    # Align images
    aligned_frames = align.process(frames)

    # Retreive images
    color_frame = aligned_frames.get_color_frame()

    # Convert to arrays
    color = numpy.asanyarray(color_frame.get_data())

    # Write as image
    cv2.imwrite('webserver/tmp/image2.jpg', color)
    time.sleep(0.01)

    # Print
    print("Demo 2 - sawblade - running")

    # Sleep
    time.sleep(0.01)

    # Show image
    #cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
    #cv2.imshow('Data generation', color)
    #k  = cv2.waitKey(1)