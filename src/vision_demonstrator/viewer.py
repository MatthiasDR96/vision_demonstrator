# Imports
import cv2
import numpy as np

# Draw reference axis
def draw_axes(img, mtx, dist, rvecs, tvecs, length):

    # Frame axis
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]])
    
    # Project 3D points in 2D
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.astype(int)

    # Line projections
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 5)
    text_pos = (imgpts[1].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'X', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    text_pos = (imgpts[2].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Y', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    text_pos = (imgpts[3].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Z', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

    return img

# Draw box
def draw_cube(img, mtx, dist, rvecs, tvecs, x, y, z, size):

    # Frame axis
    box = np.float32([[x, y, z], [x+size, y, z], [x+size, y+size, z], [x, y+size, z],
                        [x, y, z-size], [x+size, y, z-size], [x+size, y+size, z-size], [x, y+size, z-size]]).reshape(-1, 3)
    
    # Project 3D points in 2D
    imgpts, _ = cv2.projectPoints(box, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.astype(int)
    
    # Line projections
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), (0, 255, 255), 3)

    return img

# Show ball, pixel, depth and XYZ
def draw_ball_pixel(image, x, y, z, r):

    # Project 3D cooridnates
    cv2.putText(image, "X: " + str(int(round(x))) + ' mm', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Y: " + str(int(round(y))) + ' mm', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Z: " + str(int(round(z))) + ' mm', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Project radius
    cv2.putText(image, "Radius in mm: " + str(int(r)), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image
