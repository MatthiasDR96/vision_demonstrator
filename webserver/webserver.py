# Imports
import cv2
import time
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template

# Create app
app = Flask(__name__)

# Params
screen_size = (1920, 1080)
delay = 0.03

# Define streams
stream = {"demo1_image": None, "demo2_image": None, "demo3_image": None, "demo4_image": None}

# Callback
def on_message(client, userdata, message):

    # Decode image
    nparr = np.frombuffer(message.payload, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    stream[message.topic] = frame

# Init MQTT broker
client = mqtt.Client()

# Connect to MQTT broker
client.connect("mqtt.eclipseprojects.io")

# Start loop
client.loop_start()
client.subscribe([("demo1_image",0),("demo2_image",0),("demo3_image",0), ("demo4_image",0)])
client.on_message=on_message

# Generate frames
def gen(stream_id):

    # Loop 
    while True:

        # Get file name
        #file = 'webserver/tmp/image' + str(stream_id) + '.jpg'

        # Read the image from the fifo
        #with open(file, 'rb') as f:
            #check_chars = f.read()[-2:]
        #if check_chars != b'\xff\xd9':
            #frame = None
        #else:
           #frame = cv2.imread(file)

        # Get frame
        frame = stream["demo" + str(stream_id) + "_image"]

        # Check if there is a frame
        if frame is None: continue

        # Resize image to fit screen
        frame_resized = cv2.resize(frame, screen_size)     

        # Encode the frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame_resized)

        # Check if encoding went well
        if not ret: continue

        # Convert image to bytes
        frame_bytes = jpeg.tobytes()

        # Yield the frame for display in the app
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Main route
@app.route('/')
def home():
    return render_template('home.html')

# Route to streams
@app.route("/video_feed/<int:stream_id>")
def video_feed(stream_id):
    return Response(gen(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# Main
if __name__ == '__main__':

    # Run app
    app.run(host='0.0.0.0', debug=True)