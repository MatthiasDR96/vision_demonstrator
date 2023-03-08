# Imports
import cv2
import numpy as np

# Extrinsic calibration
def extrinsic_calibration(img, mtx, dist):

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # World coordinates in chessboard
    objp = np.zeros((9 * 14, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:14].T.reshape(-1, 2)
    objp = 17.4 * objp

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get chessboard corners
    ret, corners = cv2.findChessboardCornersSB(gray, (9, 14), cv2.CALIB_CB_MARKER)

    # If corners are found
    if ret == True:
        
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Extrinsic calibration
        ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # Get extrinsic matrix
        rvecs_matrix = cv2.Rodrigues(rvecs)[0]
        extrinsics = np.hstack((rvecs_matrix, tvecs))
        extrinsics = np.vstack((extrinsics, [0.0, 0.0, 0.0, 1.0]))

        return ret, corners2, rvecs, tvecs, extrinsics

    # If corners not found
    else:
        return None, None, None, None, None

# Covert 2D to 3D cooridnates
def intrinsic_trans(pixel, z, mtx):
    if (z):
        x = (pixel[0] - mtx[0, 2]) / mtx[0, 0] * z
        y = (pixel[1] - mtx[1, 2]) / mtx[1, 1] * z
        return x, y, z
    else:
        return None, None, None

# Convert camera to world coordinates
def extrinsic_trans(depth, x, y, z, ext):
    if (depth):
        mat = np.array([[x], [y], [z], [1]])
        inv = np.linalg.inv(ext)
        world = np.dot(inv, mat)
        xw, yw, zw = world[0, 0], world[1, 0], world[2, 0],
        newx = yw
        newy = xw
        newz = -zw
        return newx, newy, newz
    else:
        return None, None, None