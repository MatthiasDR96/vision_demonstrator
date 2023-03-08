# Imports
import cv2
import time
import yaml
from vision_demonstrator.Camera import Camera

# Load params
with open("config/demo4_config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Create camera object
cam = Camera("RealSense", config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Loop
while True:

    # Get start time
    t1 = time.time()

    # Read frame
    color_image, depth_image = cam.read()

    ### End of loop

    # Write as image
    cv2.imwrite('webserver/tmp/image4.jpg', color_image)

    # Print
    print("Demo 4 - classification - running")

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))


