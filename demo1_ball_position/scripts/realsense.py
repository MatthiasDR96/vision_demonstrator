# Imports
import cv2
import numpy
import pyrealsense2 as realsense

# Params
color_resolution = (1920, 1080)
depth_resolution = (1280, 720)
frames_per_second = 30
id = '821312060313'

# Connect
conn = realsense.pipeline()

# Config
conf = realsense.config()
conf.enable_device(id)
conf.enable_stream(realsense.stream.depth, depth_resolution[0], depth_resolution[1], realsense.format.z16, frames_per_second)
conf.enable_stream(realsense.stream.color, color_resolution[0], color_resolution[1], realsense.format.bgr8, frames_per_second)
		
# Start streaming
conn.start(conf)

# Align images
align = realsense.align(realsense.stream.color)

# Wait for image
frames = conn.wait_for_frames()

# Align images
aligned_frames = align.process(frames)

# Retreive images
color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

# Convert to arrays
depth = numpy.asanyarray(depth_frame.get_data())
color = numpy.asanyarray(color_frame.get_data())

# Write as image
cv2.imwrite('image.jpg', color)

# Show image
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test', color)
cv2.waitKey(1)