# Script 2: generate and write images to /tmp/pipe1
import cv2
import time

cap = cv2.VideoCapture(0)

# Loop
while True:

    # Read frame
    success, frame = cap.read()

    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Write as image
    cv2.imwrite('webserver/tmp/pipe2/image.jpg', frame)
    
    # Write as binary
    with open('webserver/tmp/pipe2/image.raw', 'wb') as f:
        ret, frame = cv2.imencode('.jpg', frame)
        f.write(frame.tobytes())

    # Sleep
    time.sleep(0.1)