# Imports
import os
import cv2
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from vision_demonstrator.decode import *
from  vision_demonstrator.preprocessing import *

# Load data
df = pd.read_csv("data/color_data.csv")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'data/model.sav'
model = pickle.load(open(filename, 'rb'))

# Accuracy
correct = 0

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

	# Iterate over first three contours
	prediction = ''
	for j, band in enumerate(bands):

		# Predict
		pred = model.predict([band])

		# Convert to class
		prediction += labelencoder.inverse_transform(pred)[0]

	# Draw text
	if len(prediction) == 3:

		# Accuracy
		if prediction == label: correct += 1

		# Plot text
		cv2.putText(img=debug3, text=prediction + " - " + decode(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
	
	# Show
	#cv2.imshow('Data evaluation', debug3)
	#cv2.waitKey(1)

# Get last sample number
print("Accuracy: " + str(correct/len(os.listdir("data/resistor_images/"))*100) + ' %')

