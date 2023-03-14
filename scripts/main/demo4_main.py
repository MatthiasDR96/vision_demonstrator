# Imports
import cv2
import time
import yaml
import torch
import torch.nn as nn
from torchvision import models
import paho.mqtt.client as mqtt
import torchvision.transforms as transforms
from vision_demonstrator.Camera import Camera
from torchvision.models import ResNet34_Weights

# Script rate
rate = 0.2 # Seconds per loop

# Load params
with open("config/demo4_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera("RealSense", config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Get trained model
model = models.resnet34()
num_ftrs = model.fc.in_features

# Define classes
classes = ["No defect", "Nut", "Scratch", ]

# Create new classification part
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('data/model6.pth'), strict=False)

# Put model in evaluation mode
model.eval()

# Link model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model.to(device)

# Define Transforms
transform = transforms.ToTensor()

# Loop
with torch.no_grad():
	while True:

		# Get start time
		t1 = time.time()

		# Read frame
		color_image, depth_image = cam.read()

		# Transform data
		tensor = transform(color_image)
		tensor = tensor.unsqueeze(0)

		# Predict class
		output = model(tensor)

		# Print output
		#color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
		cv2.putText(color_image, classes[output.argmax()], [400, 100], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
		
		### End of loop

		# Resize image
		final_image = cv2.resize(color_image, (1080, 1920)) 

		# Publish data
		data = cv2.imencode('.jpg', final_image)[1].tobytes()
		client.publish("demo4_image", data)

		# Get end time
		t2 = time.time()

		# Sleep
		if (t2-t1) < rate: time.sleep(rate - (t2-t1))

		# Get end time
		t3 = time.time()

		# Print
		print("Demo 4 - classification - running at cycle time of " + str(t3-t1) + " seconds")


