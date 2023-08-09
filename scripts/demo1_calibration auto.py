# Imports
import cv2
import yaml
import numpy as np
from scipy import optimize 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyswarm import pso
from vision_demonstrator.Camera import Camera

# Load params
with open("config/demo1_config.yaml", 'r') as stream:
	config = yaml.safe_load(stream)

# Create camera object
cam = Camera('RealSense', config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Read data from previous calibrations
hsvfile = np.load('data/demo1_hsv.npy')

# Nonlinear contraint
def objective_function(x):

	# Convert to int
	x = np.int32(x)

	# Read images
	color_image, depth_image = cam.read()

	# Crop image
	color_image = color_image[0:900, :, :]

	# Define bounds on Hue value
	lower_color = np.array([x[0], x[2], 0])
	upper_color = np.array([x[1], x[3], 255])

	# Gaussian blur
	blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

	# Convert to hsv color space
	hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

	# Get mask
	mask = cv2.inRange(hsv, lower_color, upper_color)

	# Erode to close gaps
	kernel = np.ones((10, 10), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=2)

	# Dilate to reduce data
	kernel = np.ones((10, 10), np.uint8)
	mask = cv2.dilate(mask, kernel, iterations=2)

	# Plot
	mask = cv2.resize(mask, (int(1920/2), int(1080/2)))  
	cv2.imshow('frame', mask)
	cv2.resizeWindow("frame", (int(1920/2), int(1080/2)))  
	cv2.moveWindow("frame", int(1920/2), 0)
	cv2.waitKey(1)

	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	# Find contour with largest area
	if len(contours) == 1:
		cost = cv2.contourArea(contours[0]) - 28838.0
	else:
		cost = float('inf')

	return cost

# Nonlinear contraint
def con(x):

	# Convert to int
	x = np.int32(x)

	# Read images
	color_image, depth_image = cam.read()

	# Crop image
	color_image = color_image[0:900, :, :]

	# Define bounds on Hue value
	lower_color = np.array([x[0], x[2], 0])
	upper_color = np.array([x[1], x[3], 255])

	# Gaussian blur
	blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

	# Convert to hsv color space
	hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

	# Get mask
	mask = cv2.inRange(hsv, lower_color, upper_color)

	# Erode to close gaps
	kernel = np.ones((100, 100), np.uint8)
	mask = cv2.erode(mask, None, iterations=2)

	# Dilate to reduce data
	kernel = np.ones((100, 100), np.uint8)
	mask = cv2.dilate(mask, None, iterations=2)

	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	# Find contour with largest area
	num_cts = len(contours) 

	return num_cts - 1


# Optimize
lb = [99, 88, 0, 0]
ub = [116, 255, 255, 255]
xopt, fopt = pso(objective_function, lb, ub)

# Save data
hsvarray = np.array([xopt[0], xopt[1], xopt[2], xopt[3], 0, 255])
np.save('data/demo1_hsv.npy', hsvarray)

# Define bounds on Hue value
lower_color = np.array([xopt[0], xopt[2], 0])
upper_color = np.array([xopt[1], xopt[3], 255])

# Read images
color_image, depth_image = cam.read()

# Crop image
color_image = color_image[0:900, :, :]

# Gaussian blur
blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

# Convert to hsv color space
hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

# Get mask
mask = cv2.inRange(hsv, lower_color, upper_color)

# Erode to close gaps
mask = cv2.erode(mask, None, iterations=2)

# Dilate to reduce data
mask = cv2.dilate(mask, None, iterations=2)

# Plot
mask = cv2.resize(mask, (int(1920/2), int(1080/2)))  
cv2.imshow('frame', mask)
cv2.resizeWindow("frame", (int(1920/2), int(1080/2)))  
cv2.moveWindow("frame", int(1920/2), 0)
cv2.waitKey(0)


		 