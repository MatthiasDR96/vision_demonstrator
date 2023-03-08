# Imports
import cv2
import time
import numpy as np
import paho.mqtt.client as mqtt

# Callback
def on_message(client, userdata, message):

    # Decode image
    nparr = np.frombuffer(message.payload, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Show image
    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)

# Init MQTT broker
client = mqtt.Client()

# Connect to MQTT broker
client.connect("mqtt.eclipseprojects.io")

# Start loop
client.loop_start()
client.subscribe("demo1_image")
client.on_message=on_message 

# Stop loop
time.sleep(30)
client.loop_stop()
