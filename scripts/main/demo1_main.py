# Imports
import cv2
import time
import yaml
import numpy as np
import paho.mqtt.client as mqtt
from vision_demonstrator.viewer import *
from vision_demonstrator.Camera import Camera
from vision_demonstrator.camera_callibration import *

# Load params
with open("config/demo1_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera('RealSense', config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Get HSV calibration params 
hsvfile = np.load('data/hsv.npy')

# Loop
while True:

	# Get start time
	t1 = time.time()

	# Read frame
	color_image, depth_image = cam.read()

	# Copy colour image
	final_image = color_image.copy()

	# Crop image
	color_image = color_image[0:900, :, :]

	# Gaussian blur
	blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

	# Convert to hsv color space
	hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

	# Get mask
	mask = cv2.inRange(hsv, np.array([hsvfile[0], hsvfile[2], hsvfile[4]]), np.array([hsvfile[1], hsvfile[3], hsvfile[5]]))

	# Erode to close gaps
	mask = cv2.erode(mask, None, iterations=2)

	# Dilate to get original size
	mask = cv2.dilate(mask, None, iterations=2)

	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	# If ball is present
	depth_pixel, xcam, ycam, zcam = None, None, None, None
	if len(contours) > 0:

		# Find contour with largest area
		maxcontour = max(contours, key=cv2.contourArea)

		# Find moments of the largest contour
		moments = cv2.moments(maxcontour)

		# Find ball center with moments
		center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

		# Find radius of circle
		((x, y), radius) = cv2.minEnclosingCircle(maxcontour)

		# Get pixel depth
		depth_pixel = depth_image[center[1], center[0]]

		# Transform 2D to 3D camera coordinates
		xcam, ycam, zcam = intrinsic_trans(center, depth_pixel, cam.mtx)

		# Plot ball pixel
		cv2.circle(final_image, center, 5, (0, 0, 255), -1)
		cv2.circle(final_image, center, int(radius), (255, 0, 0), 5)
		center_as_string = ''.join(str(center))
	
	# Exctrinsic calibration
	ret, corners, rvecs, tvecs, ext = extrinsic_calibration(color_image, cam.mtx, cam.dist)

	# Draw chessboard
	if ret:

		# Transform camera coordinates to world coordinates
		yworld, xworld, zworld = extrinsic_trans(depth_pixel, xcam, ycam, zcam, ext)

		# Plot chessboard
		final_image = cv2.drawChessboardCorners(final_image, (9, 14), corners, ret)
		final_image = draw_axes(final_image, cam.mtx, cam.dist, rvecs, tvecs, 3*config['chessboard_size'])

	# Rotate image
	final_image = cv2.rotate(final_image, cv2.ROTATE_180)

	# Show ball and coordinates
	if ret and depth_pixel:
		final_image = draw_ball_pixel(final_image, xworld, yworld, zworld, radius/3.3333)

	### End of loop

	# Write as image
	# cv2.imwrite('webserver/tmp/image1.jpg', final_image)

	# Show
	#cv2.imshow("People detection", final_image)
	#if cv2.waitKey(1) > -1:
		#break

	# Publish data
	data = cv2.imencode('.jpg', final_image)[1].tobytes()
	client.publish("demo1_image", data)

	# Get end time
	t2 = time.time()

	# Sleep
	if (t2-t1) < 0.2: time.sleep(0.2 - (t2-t1))

	# Get end time
	t3 = time.time()

	# Print
	print("Demo 1 - 3D ball detection - running at cycle time of " + str(t3-t1) + " seconds")
