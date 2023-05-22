# Imports
import cv2
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
import paho.mqtt.client as mqtt
import torchvision.transforms as transforms
from vision_demonstrator.Camera import Camera

# Script rate
rate = 0.5 # Seconds per loop

# Load params
with open("config/demo4_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera("RealSense", config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Init MQTT server
client = mqtt.Client(client_id="", clean_session=True, userdata=None)
client.connect("mqtt.eclipseprojects.io")
client.max_queued_messages_set(1)

# Define normalisation params
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Transforms.compose is used to perform multiple sequential transformations on an image
data_transform = transforms.Compose([
		transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# Get trained model
model = models.resnet34()
num_ftrs = model.fc.in_features

# Define classes
classes = ["No defect", "Nut", "Scratch", ]

# Create new classification part
#model.fc = nn.Linear(num_ftrs, len(classes))
#model.load_state_dict(torch.load('data/model4.pth'), strict=False)

# Put model in evaluation mode
#model.eval()

# Link model to device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model= nn.DataParallel(model)
#model.to(device)

# Loop
with torch.no_grad():
	while True:

		# Get start time
		t1 = time.time()

		# Read frame
		image, _ = cam.read()
		
		# Transform data
		#tensor = data_transform(image)
		#tensor = tensor.unsqueeze(0)

		# Predict class
		#output = model(tensor)

		# Print output
		#color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
		#cv2.putText(image, classes[output.argmax()], [400, 100], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
		cv2.putText(image, 'Knot', [400, 100], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
		
		### End of loop

		# Display the resulting frame
		#cv2.imshow('frame', image)
		#if cv2.waitKey(10) & 0xFF == ord('q'):
			#break

		cv2.imwrite('./data/demo4/image1.jpg', image)

		# Publish data
		data = cv2.imencode('.jpg', image)[1].tobytes()
		client.publish("demo4_image", data)

		# Get end time
		t2 = time.time()

		# Sleep
		if (t2-t1) < rate: time.sleep(rate - (t2-t1))

		# Get end time
		t3 = time.time()

		# Print
		print("Demo 4 - classification - running at cycle time of " + str(t3-t1) + " seconds")


