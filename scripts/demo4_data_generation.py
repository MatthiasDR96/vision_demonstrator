# Imports
import os
import cv2
import yaml
from vision_demonstrator.Camera2 import Camera

# SET THE NUMBER OF SAMPLES PER CLASS
number_of_samples_per_class = 100

# Folders
path_class_nut = "data/nut"
path_class_small_scratch = "data/small_scratch"
path_class_large_scratch = "data/large_scratch"
path_class_ok = "data/no_defect"

# Create folders
if not os.path.exists(path_class_nut): os.makedirs(path_class_nut)
if not os.path.exists(path_class_small_scratch): os.makedirs(path_class_small_scratch)
if not os.path.exists(path_class_large_scratch): os.makedirs(path_class_large_scratch)
if not os.path.exists(path_class_ok): os.makedirs(path_class_ok)

# Load params
with open("config/demo4_config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Create camera object
cam = Camera(config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])
print(cam)

# Data generation loop
counter = 1
print("\nData collection started.")
print("\nPress space bar to save a frame from the camera (click on image window)")
for i in os.listdir("data/"): # For all classes
    print("\nStart image collection of samples with " + i)
    while counter <= number_of_samples_per_class: # For all samples

        # Define path
        path = "data/" + i + "/sample_" + str(counter) + ".jpg"

        # Read frame
        color_image, depth_image = cam.read()

        # Show image
        cv2.namedWindow('Data generation', cv2.WINDOW_NORMAL)
        cv2.imshow('Data generation', color_image)
        k  = cv2.waitKey(1)

        # If space bar is pressed
        if k%256  == 32:
            cv2.imwrite(path, color_image)
            print('\tImage saved to ' + path)
            counter += 1

    # Reset counter for next class
    counter = 1

    # Print
    print("Stop image collection of samples with " + i)

# Exit
print("\nData collection ended.\n")
exit(0)



