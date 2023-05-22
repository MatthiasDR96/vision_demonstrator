# Imports
import cv2
import time
import yaml
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

# Init MQTT server
client = mqtt.Client(client_id="", clean_session=True, userdata=None)
client.connect("mqtt.eclipseprojects.io")
client.max_queued_messages_set(1)

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

	# Extract resistor bounding box
	ret, rect, debug1 = extract_resistor(image)

	# Extract cropped bands
	ret, crop, debug2 = align_resistor(image, rect)

	# Extract color band contours
	ret, bands, debug3 = extract_color_bands(debug1, crop)

	# Iterate over first three contours
	prediction = ''
	for band in bands:

		# Predict
		pred = model.predict([band])

		# Convert to class
		prediction += labelencoder.inverse_transform(pred)[0]

	# Draw text
	if len(prediction) == 3: cv2.putText(debug3, text=prediction + " - " + decode(prediction), org=(550, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

	### End of loop

	# Display the resulting frame
	#cv2.imshow('frame', debug3)
	#if cv2.waitKey(10) & 0xFF == ord('q'):
		#break 

	# Publish data
	data = cv2.imencode('.jpg', debug3)[1].tobytes()
	client.publish("demo3_image", data)

	# Get end time
	t2 = time.time()

	# Sleep
	if (t2-t1) < rate: time.sleep(rate - (t2-t1))

	# Get end time
	t3 = time.time()

	# Print
	print("Demo 3 - resistors - running at cycle time of " + str(t3-t1) + " seconds")
