# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_resistor(image):

	# Convert to RGB
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Convert to gray, and threshold
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Threshold background
	_, threshed = cv2.threshold(image_gray, 230, 255, cv2.THRESH_BINARY_INV)

	# Morphological transformations to remove sticks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
	morphed_open = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
	morphed_close = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel)

	# Find contour of resistor
	contours = cv2.findContours(morphed_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Check if there are contours
	if len(contours) == 0: return False, None

	# Get largest contour
	maxcontour = max(contours, key=cv2.contourArea)

	# Get minimal area rectangle
	rect = cv2.minAreaRect(maxcontour)

	# Get rectangle properties
	angle = rect[2]
	rows, cols = image.shape[0], image.shape[1]

	# Rotate image
	M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]), angle-90, 1)
	img_rot = cv2.warpAffine(image,M,(cols,rows))

	# Rotate bounding box 
	box = cv2.boxPoints((rect[0], rect[1], angle))
	pts = np.intp(cv2.transform(np.array([box]), M))[0]    
	pts[pts < 0] = 0

	# Cropping
	cropped = img_rot[pts[0][1]+20:pts[3][1]-100, pts[0][0]+40:pts[2][0]-140]

	# Check if cropped image has some rows
	if cropped.shape[1] == 0: return False, None

	# Bilateral filtering
	cropped = cv2.bilateralFilter(cropped, 15, 35, 35)

	return True, cropped

def extract_color_bands(image):

	# Remove area in between color bands
	mask = cv2.bitwise_not(cv2.inRange(image, np.array([120, 120, 110]), np.array([190, 190, 180])))

	# Find the three largest contours of the color bands
	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Get three largest contours
	largest_contours = sorted(contours, key=cv2.contourArea)
	largest_contours.reverse()
	largest_contours = largest_contours[0:3]

	# Plot
	tmp = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 1)

	# Sort contours from left to right
	sorted_contours = sorted(largest_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
	sorted_contours.reverse()

	# Make sure there are three bands
	if len(sorted_contours) < 3:
		return False, None

	# Check if big enough
	if cv2.contourArea(largest_contours[2]) < 50:
		return False, None

	return True, sorted_contours