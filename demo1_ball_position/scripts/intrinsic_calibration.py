# Imports
import cv2 as opencv
import numpy
from demo1_ball_position.Camera import Camera
from demo1_ball_position.Viewer import Viewer

# Window settings
window_name = 'Intrinsic calibration'
window_b, window_h = 1920, 1080

# Get camera object
cam = Camera()

# Get Viewer object
viewer = Viewer()

# Start camera
cam.start()

# termination criteria max 30 iteraties of epsilon(nauwkeurigheid) van 0,001
criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# bereid de objectpunten van het schaakbordpatroon voor
objp = numpy.zeros((cam.h * cam.b, 3), numpy.float32)
objp[:, :2] = numpy.mgrid[0:cam.b, 0:cam.h].T.reshape(-1, 2)
objp = cam.size * objp

# wereldpunten en beeldpunten verzamelen in arrays
objpoints = []  # 3d punten in de ruimte
imgpoints = []  # corresponderende punten in 2d

# Loop
k = 0
while k < 30:

    # Wait
    print("Shot")
    opencv.waitKey(1000)

    # Read images
    color_image, depth_image = cam.read()

    # To grayscale
    gray_image = opencv.cvtColor(color_image, opencv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = opencv.findChessboardCorners(gray_image, (cam.b, cam.h), None)
    
    # If corners found
    if ret == True:

        # Refine corners
        corners2 = opencv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

        # Append
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Show chessbaord corners
        color_image = opencv.drawChessboardCorners(color_image, (cam.b, cam.h), corners2, ret)

        # Iter
        k += 1

    # Show image
    viewer.show_image(color_image)

# End calibration
opencv.destroyAllWindows()

# Calculate intrinsics
ret, mtx, dist, rvecs, tvecs = opencv.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
print("intrinsieke matrix:\n")
print(mtx)
print("distortie:\n")
print(dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = opencv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = opencv.norm(imgpoints[i], imgpoints2, opencv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("\ntotal error:\n", mean_error / len(objpoints))

# Save
numpy.save('intrinsics1.npy', mtx)

'''
opslaan in array niet uitgewerkt omdat dit project werkt met de D415.
de intrinsieke matrix hiervan werd bepaald met de fabrieksgegevens:
intrinsieke matrix:
[[1.34e+03 0.00e+00 9.60e+02]
 [0.00e+00 1.34e+03 5.40e+02]
 [0.00e+00 0.00e+00 1.00e+00]]
distorsie verwaarloosbaar:
[0. 0. 0. 0. 0.]
dit script komt ongeveer dezelfde waarden uit
'''