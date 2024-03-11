# Imports
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Camera params
color_resolution = (1920, 1080)
depth_resolution = (1280, 720)
frames_per_second = 30

# Connect to realsense
pipeline = rs.pipeline()

# Config camera
config = rs.config()
config.enable_device('821312060313')
config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frames_per_second)
config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, frames_per_second)

# Start streaming
pipeline.start(config)

# Loop
while True:

	# Get RGB frame from camera
	frames = pipeline.wait_for_frames()
	frame = frames.get_color_frame()
	frame = np.asanyarray(frame.get_data())
	
	# Run batched inference on a list of images
	results = model(frame, stream=True)  # return a list of Results objects

	# Process results list
	for result in results:

		# Plot results
		result_tmp = result.plot()  # display to screen

		# Resize image
		screensize = 4096, 2160
		resized = cv2.resize(result_tmp, screensize, interpolation=cv2.INTER_AREA)
		resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
		
		# Show
		cv2.imshow("People detection", resized)
		if cv2.waitKey(1) > -1:
			break




