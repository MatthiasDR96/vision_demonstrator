# Imports
import cv2
import time
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template

# Create app
app = Flask(__name__)

# Params
screen_size = (1920, 1080-120)

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

		# Get frame s
		frame1 = stream["demo1_image"]
		frame2 = stream["demo2_image"]
		frame3 = stream["demo3_image"]
		frame4 = stream["demo4_image"]

		# Check if there is a frame
		if frame1 is None: frame1 = np.zeros((screen_size[0],screen_size[1],3), np.uint8)
		if frame2 is None: frame2 = np.zeros((screen_size[0],screen_size[1],3), np.uint8)
		if frame3 is None: frame3 = np.zeros((screen_size[0],screen_size[1],3), np.uint8)
		if frame4 is None: frame4 = np.zeros((screen_size[0],screen_size[1],3), np.uint8)

		# Concat frames
		row1 = np.hstack([frame1, frame2])
		row2 = np.hstack([frame3, frame4])
		frame = np.vstack([row1, row2])

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
	app.run(host='0.0.0.0', debug=False)