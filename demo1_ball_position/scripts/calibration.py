# Imports
import os
import cv2
import numpy
from demo1_ball_position.Camera import Camera
from demo1_ball_position.Viewer import Viewer

# Read data from previous calibrations
hsvfile = numpy.load('demo1_ball_position/data/hsv.npy')

# Get camera object
cam = Camera()
cam.start()

# Get Viewer object
viewer = Viewer()

def nothing(*args):
    pass

# Make sliders
cv2.createTrackbar('Hmin', viewer.window_name, hsvfile[0], 179, nothing)
cv2.createTrackbar('Hmax', viewer.window_name, hsvfile[1], 179, nothing)
cv2.createTrackbar('Smin', viewer.window_name, hsvfile[2], 255, nothing)
cv2.createTrackbar('Smax', viewer.window_name, hsvfile[3], 255, nothing)
cv2.createTrackbar('Vmin', viewer.window_name, hsvfile[4], 255, nothing)
cv2.createTrackbar('Vmax', viewer.window_name, hsvfile[5], 255, nothing)
cv2.createTrackbar('save', viewer.window_name, 0, 1, nothing)

# Define image formats
HSVmin = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
HSVmax = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
HSVgem = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
white_image = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)

# Initial mask
white_image[:] = [255, 255, 255]

# Loop
while True:

    # Get slider values
    hmin = cv2.getTrackbarPos('Hmin', viewer.window_name)
    hmax = cv2.getTrackbarPos('Hmax', viewer.window_name)
    smin = cv2.getTrackbarPos('Smin', viewer.window_name)
    smax = cv2.getTrackbarPos('Smax', viewer.window_name)
    vmin = cv2.getTrackbarPos('Vmin', viewer.window_name)
    vmax = cv2.getTrackbarPos('Vmax', viewer.window_name)
    save = cv2.getTrackbarPos('save', viewer.window_name)

    # Read images
    color_image, depth_image = cam.read()

    # Define bounds on Hue value
    lower_color = numpy.array([hmin, smin, vmin])
    upper_color = numpy.array([hmax, smax, vmax])

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

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Binary of image
    mask_bgr = cv2.bitwise_and(white_image, white_image, mask=mask)

    # Mount all images
    img = numpy.hstack((color_image, mask_bgr, res))

    # Show images
    viewer.show_image(img)

    # Leave loop on save button
    if (save): break

# Save data
hsvarray = numpy.array([hmin, hmax, smin, smax, vmin, vmax])
numpy.save('demo1_ball_position/data/hsv.npy', hsvarray)