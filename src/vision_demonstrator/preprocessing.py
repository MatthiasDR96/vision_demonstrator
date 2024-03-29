# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_resistor(image):

	# Convert to gray to threshold background
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Threshold background
	_, threshed = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)

	cv2.imwrite("./data/demo3/image1.jpg", threshed)

	# Morphological transformations to remove sticks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
	morphed_open = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
	image_thresh = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel)

	cv2.imwrite("./data/demo3/image2.jpg", image_thresh)

	# Find contour of resistor
	contours = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Filter contours
	contours = list(filter(lambda x: (cv2.contourArea(x) < 20000), contours)) 
	contours = list(filter(lambda x: (cv2.contourArea(x) > 5000), contours)) 

	# Check if there are contours
	if len(contours) == 0: return False, None, image

	# Get largest contour
	maxcontour = max(contours, key=cv2.contourArea)

	# Get minimal area rectangle
	rect = cv2.minAreaRect(maxcontour)

	# Draw rect
	box = np.int0(cv2.boxPoints(rect))
	debug_image = cv2.drawContours(image.copy(),[box], 0, (0,0,255), 2)

	return True, rect, debug_image

def align_resistor(image, rect):

	# Check if input is okay
	if rect is None: return False, None, image

	# Get rectangle properties
	angle = rect[2]
	rows, cols = image.shape[0], image.shape[1]

	# Make sure alignment is horizontaly
	#print(float(rect[1][0])/rect[1][1])
	#if float(rect[1][0])/rect[1][1] < 1:
		#rotation = angle + 90
	#else:
		#rotation = 90 -(angle - 90)

	# Rotate image
	M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]), angle-90, 1)
	image_aligned = cv2.warpAffine(image,M,(cols,rows))

	# Rotate bounding box 
	box = cv2.boxPoints((rect[0], rect[1], angle))
	pts = np.intp(cv2.transform(np.array([box]), M))[0]    
	pts[pts < 0] = 0

	# Cropping
	image_cropped = image_aligned[pts[0][1]:pts[3][1], pts[0][0]:pts[2][0]]
	
	# Check if cropped image has some rows and columns
	if image_cropped.shape[1] == 0 or image_cropped.shape[0] == 0: return False, None, image_aligned

	# Bilateral filtering
	image_cropped = cv2.bilateralFilter(image_cropped, 15, 35, 35)
	
	return True, image_cropped, image_aligned

def extract_color_bands(image, crop):

	# Check if input is okay
	if crop is None: return False, [], image

	# Get HSV calibration params 
	hsvfile1 = np.load('data/demo3_hsv_resistor.npy')
	hsvfile2 = np.load('data/demo3_hsv_background.npy')

	# Convert image to HSV
	hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

	# Remove area in between color bands
	mask1 = cv2.bitwise_not(cv2.inRange(hsv, np.array([hsvfile1[0], hsvfile1[2], hsvfile1[4]]), np.array([hsvfile1[1], hsvfile1[3], hsvfile1[5]])))
	mask2 = cv2.inRange(hsv, np.array([hsvfile2[0], hsvfile2[2], hsvfile2[4]]), np.array([hsvfile2[1], hsvfile2[3], hsvfile2[5]]))
	mask = cv2.bitwise_and(mask1, mask2)

	cv2.imwrite("./data/demo3/image5.jpg", mask1)
	cv2.imwrite("./data/demo3/image6.jpg", mask2)
	cv2.imwrite("./data/demo3/image7.jpg", mask)

	# Morphological transformations to remove sticks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	morphed_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel)

	cv2.imwrite("./data/demo3/image8.jpg", mask)
	
	# Find the three largest contours of the color bands
	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Plot
	crop_ = cv2.drawContours(crop.copy(), contours, -1, (255,0,0), 3)

	# Debug image
	debug_image = image.copy()
	debug_image[0:crop_.shape[0], 0:crop_.shape[1]] = crop_

	# Remove contours outside ROI
	remaining_contours = []
	for cnt in contours:
		#if cv2.contourArea(cnt) < 1000: continue
		x,y,w,h = cv2.boundingRect(cnt)
		if x < 250: remaining_contours.append(cnt)

	# Check if enough contours
	if len(remaining_contours) < 3: return False, [], debug_image

	# Get three largest contours
	largest_contours = sorted(remaining_contours, key=cv2.contourArea, reverse=True)[0:3]

	# Sort contours from left to right
	sorted_contours = sorted(largest_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

	# Plot
	crop_ = cv2.drawContours(crop.copy(), sorted_contours, -1, (0,255,0), 3)

	cv2.imwrite("./data/demo3/image9.jpg", crop_)

	# Debug image
	debug_image = image.copy()
	debug_image[0:crop_.shape[0], 0:crop_.shape[1]] = crop_

	# Mask hsv
	hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

	# Iterate over three contours
	color_bands = []
	for ctr in sorted_contours:

		# Get roi
		x,y,w,h = cv2.boundingRect(ctr)
		roi = hsv[y+10:y+h-10, x+5:x+w-5]

		# Check if cropped image has some rows and columns
		if roi.shape[1] == 0 or roi.shape[0] == 0: return False, [], debug_image

		# Get hsv
		roi_h = [i for i in roi[:,:,0].ravel() if i != 0]  
		roi_s = [i for i in roi[:,:,1].ravel() if i != 0]  
		roi_v = [i for i in roi[:,:,2].ravel() if i != 0]  

		# Get means of HSV data
		mean_hsv = [np.mean(roi_h), np.mean(roi_s), np.mean(roi_v)]

		# Predict
		color_bands.append(mean_hsv)

	return True, color_bands, debug_image
	