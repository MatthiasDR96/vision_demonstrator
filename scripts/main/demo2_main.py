# Imports
import io
import cv2
import time
import socket
import numpy as np
from ftplib import FTP
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

# Script rate
rate = 0.5 # Seconds per loop

# Params
host = '169.254.122.41' #'10.5.5.100'
port = 9876

# Start measuring
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.send(str.encode("MEASURE /C"))
s.close()

# Command
command = str.encode("EIC C:\Data\RAMDisk/test\r")

# Start file transfer
ftp = FTP(host)
ftp.login()
ftp.cwd("RAMDisk")
parent = ftp.pwd()

# Loop
try:
	
	while True:

		# Get start time
		t1 = time.time()

		# Make screenshot
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((host, port))
		s.send(command)
		s.close()

		# Get images in folder
		image_paths = ftp.nlst()

		# Check if there is an image
		if len(image_paths) == 0: continue

		# Select required image
		image_path = image_paths[0]

		# Retrieve file
		r = io.BytesIO()
		try:
			ftp.retrbinary('RETR '+ image_path, r.write)
		except:
			continue

		# Publish data
		image = np.asarray(bytearray(r.getvalue()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

		# Get images in folder
		image_paths = ftp.nlst()

		# Check if there is an image and delete if exist
		try:
			ftp.delete(image_path)
		except:
			continue
			
		### End of loop

		# Display the resulting frame
		final_image = cv2.resize(image, (int(1920/2), int(1080/2))) 
		cv2.imshow('frame', final_image)
		cv2.resizeWindow("frame", (int(1920/2), int(1080/2)))  
		cv2.moveWindow("frame", int(1920/2), 0)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break 

		# Get end time
		t2 = time.time()

		# Sleep
		if (t2-t1) < rate: time.sleep(rate - (t2-t1))

		# Get end time
		t3 = time.time()

		# Print
		print("Demo 2 - sawblade - running at cycle time of " + str(t3-t1) + " seconds")

except:

	# Stop measuring
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((host, port))
	s.send(str.encode("MEASURE /E"))
	s.close()