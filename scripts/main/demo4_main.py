# Imports
import cv2
import time
import yaml
import paho.mqtt.client as mqtt
from vision_demonstrator.Camera import Camera

# Load params
with open("config/demo4_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera("RealSense", config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Loop
while True:

    # Get start time
    t1 = time.time()

    # Read frame
    color_image, depth_image = cam.read()

    ### End of loop

    # Write as image
    # cv2.imwrite('webserver/tmp/image4.jpg', color_image)

    # Publish data
    data = cv2.imencode('.jpg', color_image)[1].tobytes()
    client.publish("demo4_image", data)

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))

    # Get end time
    t3 = time.time()

    # Print
    print("Demo 4 - classification - running at cycle time of " + str(t3-t1) + " seconds")


