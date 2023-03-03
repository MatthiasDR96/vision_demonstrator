# Imports
import os
import cv2
import time
import config
import numpy as np
from flask import Flask, Response, render_template

# Create app
app = Flask(__name__)

# Generate frames
def gen(stream_id):
     while True:
                
        # Read the image from the fifo
        file = 'webserver/tmp/image' + str(stream_id) + '.jpg'
        #with open(file, 'rb') as f:
            #check_chars = f.read()[-2:]
            #if check_chars != b'\xff\xd9':
                #print('Not complete image')
            #else:
        frame = cv2.imread(file)

        # Check if there is a frame
        if frame is None: continue

        # Resize image to fit screen
        frame = cv2.resize(frame, config.screen_size)     

        # Encode the frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Check if encoding went well
        if not ret: continue

        # Convert image to bytes
        frame = jpeg.tobytes()

        # Yield the frame for display in the app
        time.sleep(config.deltat)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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