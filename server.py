# imports
import cv2
import paho.mqtt.client as mqtt 

# Init MQTT broker
client = mqtt.Client()

# Connect to MQTT broker
client.connect("mqtt.eclipseprojects.io")

# Read frames
vc = cv2.VideoCapture(0)

# Loop
while True:

    # Read frame
    rval, frame = vc.read()

    # Encode frame
    data = cv2.imencode('.jpg', frame)[1].tobytes()

    # Publish data
    client.publish("demo1_image", data)
