# Imports
import cv2
import config
import imutils
import numpy as np
from collections import deque
from demo1_ball_position.Camera import Camera
from demo1_ball_position.Viewer import Viewer

# Create camera object and start camera
cam = Camera()
cam.start()

# Create viewer object and set window
viewer = Viewer()

# Get HSV calibration params 
hsvfile = np.load('demo1_ball_position/data/hsv.npy')

# Get HSV calibration params
lower_color = np.array([hsvfile[0], hsvfile[2], hsvfile[4]])
upper_color = np.array([hsvfile[1], hsvfile[3], hsvfile[5]])

# Loop
while True:

    # Read frame
    color_image, depth_image = cam.read()

    # Copy colour image
    final_image = color_image.copy()

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Get mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to get original size
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # If ball is present
    depth_pixel, xcam, ycam, zcam = None, None, None, None
    if len(contours) > 0:

        # Find contour with largest area
        maxcontour = max(contours, key=cv2.contourArea)

        # Find moments of the largest contour
        moments = cv2.moments(maxcontour)

        # Find ball center with moments
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        # Find radius of circle
        ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)

        # If radius is big enough, it is the ball
        if radius >= config.ball_radius:
                
            # Get pixel depth
            depth_pixel = depth_image[center[1], center[0]]

            print(depth_pixel)

            # Transform 2D to 3D camera coordinates
            print(cam.mtx)
            xcam, ycam, zcam = cam.intrinsic_trans(center, depth_pixel, cam.mtx)

            # Plot ball pixel
            cv2.circle(final_image, center, 5, (0, 0, 255), -1)
            cv2.circle(final_image, center, int(radius), (255, 0, 0), 5)
            center_as_string = ''.join(str(center))
        
    # Exctrinsic calibration
    ret, corners, rvecs, tvecs, ext = cam.extrinsic_calibration(color_image)
    print(ext)

    # Draw chessboard
    if ret:

        # Transform camera coordinates to world coordinates
        yworld, xworld, zworld = cam.extrinsic_trans(depth_pixel, xcam, ycam, zcam, ext)

        # Plot chessboard
        final_image = cv2.drawChessboardCorners(final_image, (cam.b, cam.h), corners, ret)
        final_image = viewer.draw_axes(final_image, cam.mtx, cam.dist, rvecs, tvecs, 3*config.chessboard_size)

    # Rotate image
    final_image = cv2.rotate(final_image, cv2.ROTATE_180)

    # Show ball and coordinates
    if ret and depth_pixel:
        final_image = viewer.draw_ball_pixel(final_image, xworld, yworld, zworld, radius/3.3333)

    # Write as image
    cv2.imwrite('webserver/tmp/pipe1/image.jpg', final_image)

    # Show image
    viewer.show_image(final_image)