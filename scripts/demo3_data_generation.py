# Imports
import os
import cv2
from vision_demonstrator.decode import *
from vision_demonstrator.Camera import Camera
from vision_demonstrator.preprocessing import *

# SET THE NUMBER OF SAMPLES PER CLASS
number_of_samples_per_class = 100

# Folders
path_data = "data/resistor_images/"

# Create folders
if not os.path.exists(path_data): os.makedirs(path_data)

# Create camera object
cam = Camera('Basler', 0, 0, 0, 0)

# Data generation loop
counter = 1
print("\nData collection started.")
print("\nPress space bar to save a frame from the camera (click on image window)")
while counter <= number_of_samples_per_class: # For all samples

    # Read frame
    image, _ = cam.read()

    # Show image
    #cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
    #cv2.imshow('Data generation', image)
    #k  = cv2.waitKey(1)

    # Get user input
    label = input("Enter label:")
    
    # Define path
    file_name = str(counter) + "_" + label + ".jpg"

    # Save image
    cv2.imwrite(path_data + file_name, image)
    print('\tImage saved to ' + path_data + file_name)
    counter += 1

# Exit
print("\nData collection ended.\n")
exit(0)