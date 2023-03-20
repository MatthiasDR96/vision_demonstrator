# Imports
import os
import cv2
from vision_demonstrator.decode import *
from vision_demonstrator.Camera import Camera
from vision_demonstrator.preprocessing import *

# Folders
path_data = "data/resistor_images/"

# Create folders
if not os.path.exists(path_data): os.makedirs(path_data)

# Get last sample number
#arr = sorted(os.listdir("data/resistor_images/"), key=lambda str: int(str.split('_')[0]))
#last_sample_number = int(arr[-1].split('_')[0])

# Create camera object
cam = Camera('IDS', 0, 0, 0, 0)

# Data generation loop
counter = 1 #last_sample_number + 1
print("\nData collection started.")
print("\nPress space bar to save a frame from the camera (click on image window)")
while True: # For all samples

    # Get user input
    label = input("Enter label:")

    # Take 10 per resistor
    for _ in range(10):

        # Read frame
        image, _ = cam.read()

        # Show image
        #cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
        #cv2.imshow('Data generation', image)
        #k  = cv2.waitKey(1)

        # Define path
        file_name = str(counter) + "_" + label + ".jpg"

        # Save image
        cv2.imwrite(path_data + file_name, image)
        print('\tImage saved to ' + path_data + file_name)
        counter += 1