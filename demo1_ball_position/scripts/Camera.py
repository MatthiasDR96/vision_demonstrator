# Imports
import cv2
import numpy
import config
import pyrealsense2 as realsense

# Camera class
class Camera:

	def __init__(self):

		# Camera configuration properties
		self.color_resolution = config.color_resolution
		self.depth_resolution = config.depth_resolution
		self.frames_per_second = config.frames_per_second
		self.id = config.id

		# Camera connection properties
		self.conn = None
		self.conf = None
		self.align = None

		# Camera calibration properties
		self.mtx = numpy.load('demo1_ball_position/data/intrinsics.npy')
		self.dist = numpy.load('demo1_ball_position/data/distortion.npy')

		# Chessboard properties
		self.h = config.chessboard_height
		self.b = config.chessboard_width
		self.size = config.chessboard_size

	def start(self):

		# Connect
		self.conn = realsense.pipeline()

		# Config
		self.conf = realsense.config()
		self.conf.enable_device(self.id)
		self.conf.enable_stream(realsense.stream.depth, self.depth_resolution[0], self.depth_resolution[1], realsense.format.z16, self.frames_per_second)
		self.conf.enable_stream(realsense.stream.color, self.color_resolution[0], self.color_resolution[1], realsense.format.bgr8, self.frames_per_second)
		
		# Start streaming
		self.conn.start(self.conf)

		# Align images
		self.align = realsense.align(realsense.stream.color)

	def read(self):

		# Wait for image
		frames = self.conn.wait_for_frames()

		# Align images
		aligned_frames = self.align.process(frames)

		# Retreive images
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		# Convert to arrays
		depth = numpy.asanyarray(depth_frame.get_data())
		color = numpy.asanyarray(color_frame.get_data())
		
		return color, depth

	# Extrinsic calibration
	def extrinsic_calibration(self, img):

		# Termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# World coordinates in chessboard
		objp = numpy.zeros((self.b * self.h, 3), numpy.float32)
		objp[:, :2] = numpy.mgrid[0:self.b, 0:self.h].T.reshape(-1, 2)
		objp = self.size * objp

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Get chessboard corners
		ret, corners = cv2.findChessboardCornersSB(gray, (self.b, self.h), cv2.CALIB_CB_MARKER)
	
		# If corners are found
		if ret == True:
			
			# Refine corners
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			# Extrinsic calibration
			ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)

			# Get extrinsic matrix
			rvecs_matrix = cv2.Rodrigues(rvecs)[0]
			extrinsics = numpy.hstack((rvecs_matrix, tvecs))
			extrinsics = numpy.vstack((extrinsics, [0.0, 0.0, 0.0, 1.0]))

			return ret, corners2, rvecs, tvecs, extrinsics

		# If corners not found
		else:
			return None, None, None, None, None

	# Covert 2D to 3D cooridnates
	def intrinsic_trans(self, pixel, z, mtx):
		if (z):
			x = (pixel[0] - mtx[0, 2]) / mtx[0, 0] * z
			y = (pixel[1] - mtx[1, 2]) / mtx[1, 1] * z
			return x, y, z
		else:
			return None, None, None

	# Convert camera to world coordinates
	def extrinsic_trans(self, depth, x, y, z, ext):
		if (depth):
			mat = numpy.array([[x], [y], [z], [1]])
			inv = numpy.linalg.inv(ext)
			world = numpy.dot(inv, mat)
			xw, yw, zw = world[0, 0], world[1, 0], world[2, 0],
			newx = yw
			newy = xw
			newz = -zw
			return newx, newy, newz
		else:
			return None, None, None