# Script 1: generate and write images to /tmp
import cv2
import time

cap = cv2.VideoCapture(0)

# Loop
while True:

    # Read frame
    success, frame = cap.read()

    # Write as image
    cv2.imwrite('tmp/image3.jpg', frame)

    # Sleep
    time.sleep(0.01)