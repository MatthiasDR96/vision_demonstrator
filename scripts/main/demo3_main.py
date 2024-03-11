# Imports
import cv2
import time
import pickle
import pandas as pd
import paho.mqtt.client as mqtt
from vision_demonstrator.decode import *
from vision_demonstrator.Camera import Camera
from vision_demonstrator.preprocessing import *
from sklearn.preprocessing import LabelEncoder 

# Script rate
rate = 0.5 # Seconds per loop

# Create camera object
cam = Camera('IDS', 0, 0, 0, 0)

# Load data
df = pd.read_csv("data/color_data.csv")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'data/model.sav'
model = pickle.load(open(filename, 'rb'))

# Loop
while True:

	# Get start time
	t1 = time.time()

	# Get frame
	image, _ = cam.read()

	cv2.imwrite("./data/demo3/image_raw.jpg", image)

	# Extract resistor bounding box
	ret, rect, debug1 = extract_resistor(image)

	cv2.imwrite("./data/demo3/image_mask_bg.jpg", debug1)

	# Extract cropped bands
	ret, crop, debug2 = align_resistor(image, rect)

	cv2.imwrite("./data/demo3/image_aligned.jpg", debug2)

	# Extract color band contours
	ret, bands, debug3 = extract_color_bands(debug1, crop)

	cv2.imwrite("./data/demo3/image_contours.jpg", debug3)

	# Iterate over first three contours
	prediction = ''
	for band in bands:

		# Predict
		try:
			pred = model.predict([band])
		except:
			pass

		# Convert to class
		prediction += labelencoder.inverse_transform(pred)[0]

	# Draw text
	if len(prediction) == 3: cv2.putText(debug3, text=prediction + " - " + decode(prediction), org=(550, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

	### End of loop

	# Display the resulting frame
	final_image = cv2.resize(debug3, (int(1920/2), int(1080/2))) 
	cv2.imshow('frame', final_image)
	cv2.resizeWindow("frame", (int(1920/2), int(1080/2)))  
	cv2.moveWindow("frame", 0, int(1080/2))
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break 

	cv2.imwrite("./data/demo3/image_final.jpg", final_image)

	# Get end time
	t2 = time.time()

	# Sleep
	if (t2-t1) < rate: time.sleep(rate - (t2-t1))

	# Get end time
	t3 = time.time()

	# Print
	print("Demo 3 - resistors - running at cycle time of " + str(t3-t1) + " seconds")
