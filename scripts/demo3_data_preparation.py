# Imports
import cv2
import glob
import pandas as pd
import numpy as np
from vision_demonstrator.preprocessing import *

# Create empty dataset
df = pd.DataFrame({'H': pd.Series(dtype='int'),
				   'S': pd.Series(dtype='int'),
				   'V': pd.Series(dtype='int'),
				   'Class': pd.Series(dtype='str')})

# Loop over every image
for i in glob.glob('data/resistor_images/*jpg'):
	
	# Read image
	image = cv2.imread(i)

	# Read label
	label = i.split('_')[-1][0:3]

	# Extract resistor bounding box
	ret, rect, debug1 = extract_resistor(image)

	# Extract cropped bands
	ret, crop, debug2 = align_resistor(image, rect)

	# Extract color band contours
	ret, bands, debug3 = extract_color_bands(debug1, crop)

	# Plot
	#cv2.imshow('Data preparation', debug3)
	#cv2.waitKey(1)

	# Iterate over color bands
	for j, band in enumerate(bands):

		# New entry
		df2 = pd.DataFrame([[band[0], band[1], band[2], label[j]]], columns=['H','S','V', 'Class'])

		# Append to main dataframe
		df = pd.concat([df, df2])

# Save data
df.to_csv('data/color_data.csv', index=False)


