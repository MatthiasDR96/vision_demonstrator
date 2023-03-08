# Imports
import cv2
import time
from flask import Flask, Response, render_template
from vision_demonstrator.demo1_main import demo1
from vision_demonstrator.demo2_main import demo2
from vision_demonstrator.demo3_main import demo3

# Create app
app = Flask(__name__)

# Params
screen_size = (1920, 1080)
delay = 0.03

#demo1 = demo1()
#demo1 = demo2()
#demo1 = demo3()

# Generate frames
def gen(stream_id):

    # Loop
    while True:

        #if int(stream_id) == 1:
            #frame = demo1.read()
            #print(frame)
        #elif int(stream_id) == 1:
            #frame = demo2.read()
            #print(frame)
        #elif int(stream_id) == 1:
            #frame = demo3.read()
            #print(frame)

        # GEt file name
        file = 'webserver/tmp/image' + str(stream_id) + '.jpg'

        # Read the image from the fifo
        with open(file, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            frame = None
        else:
           frame = cv2.imread(file)

        # Check if there is a frame
        if frame is None: continue

        # Resize image to fit screen
        frame = cv2.resize(frame, screen_size)     

        # Encode the frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Check if encoding went well
        if not ret: continue

        # Convert image to bytes
        frame = jpeg.tobytes()

        # Yield the frame for display in the app
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Sleep
        time.sleep(delay)

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