# Imports
import os
import numpy as np
import cv2
import time
from flask import Flask, Response, render_template

# Create app
app = Flask(__name__)

# Generate frames
def gen(stream_id):
     while True:
                
        # Read the image from the fifo
        frame = cv2.imread('webserver/tmp/pipe' + str(stream_id) + '/image.jpg')
        if frame is None:
            continue

        # Encode the frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the frame for display in the app
        time.sleep(0.1)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Main route
@app.route('/')
def home():
    return render_template('home.html')

# Route to streams
@app.route("/video_feed1")
def video_feed():
    return Response(gen(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed2")
def video_feed():
    return Response(gen(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed3")
def video_feed():
    return Response(gen(3), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed4")
def video_feed():
    return Response(gen(4), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# Main
if __name__ == '__main__':

    # Run app
    app.run(host='0.0.0.0', debug=True)