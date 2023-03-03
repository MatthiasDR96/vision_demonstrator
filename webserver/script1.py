# Script 1: generate and write images to /tmp/pipe1
import cv2
import time

cap = cv2.VideoCapture(0)

# Loop
while True:

    # Read frame
    success, frame = cap.read()

    # Write as image
    cv2.imwrite('webserver/tmp/pipe1/image.jpg', frame)
    
    # Write as binary
    with open('webserver/tmp/pipe1/image.raw', 'wb') as f:
        ret, frame = cv2.imencode('.jpg', frame)
        f.write(frame.tobytes())

    # Sleep
    time.sleep(0.1)