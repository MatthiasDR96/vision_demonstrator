# Imports
import cv2
import numpy
from pypylon import pylon
import pyrealsense2 as realsense
import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl

# Camera class
class Camera:

	def __init__(self, type, color_resolution, depth_resolution, frames_per_second, cam_id):
		
		# Select camera type
		if type == 'RealSense':

			# Camera
			self.cam = CameraRealsense(color_resolution, depth_resolution, frames_per_second, cam_id)

			# Camera calibration properties
			self.mtx = numpy.load('data/intrinsics.npy')
			self.dist = numpy.load('data/distortion.npy')

		elif type == "Basler":
			
			# Camera
			self.cam = CameraBasler()

		elif type == "IDS":
			
			# Camera
			self.cam = CameraIDS()

	def start(self):
		return self.cam.start()

	def stop(self):
		return self.cam.stop()

	def read(self):
		return self.cam.read()

# RealSense camera class
class CameraRealsense:

	def __init__(self, color_resolution, depth_resolution, frames_per_second, cam_id):

		# Camera configuration properties
		self.color_resolution = color_resolution
		self.depth_resolution = depth_resolution
		self.frames_per_second = frames_per_second
		self.id = cam_id

		# Camera connection properties
		self.conn = None
		self.conf = None
		self.align = None

		# Start camera
		self.start()

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

		return True

	def stop(self):
		self.conn.stop()
		return True

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
		
# Basler camera class
class CameraBasler:

	def __init__(self):

		# Get camera instance
		self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

		# Start camera 
		self.start()

	def start(self):

		# Start grabbing
		self.camera.Open()
		self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

		# Converter for opencv
		self.converter = pylon.ImageFormatConverter()
		self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
		self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

	def stop(self):
		self.camera.StopGrabbing()
		#self.camera.Close()

	def read(self):
		
		while True:
			grab = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
			if grab.GrabSucceeded():
				image = self.converter.Convert(grab)
				image = image.GetArray()  
				break
			else:
				image = None   
			grab.Release()
		return image, image
			
# IDS camera class
class CameraIDS:

	def __init__(self):

		# Get camera instance
		ids_peak.Library.Initialize()
		device_manager = ids_peak.DeviceManager.Instance()
		device_manager.Update()
		device_descriptors = device_manager.Devices()
		self.device = device_descriptors[0].OpenDevice(ids_peak.DeviceAccessType_Exclusive)
		self.remote_device_nodemap = self.device.RemoteDevice().NodeMaps()[0]

		# Start camera 
		self.start()

	def start(self):

		self.remote_device_nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
		self.remote_device_nodemap.FindNode("TriggerSource").SetCurrentEntry("Software")
		self.remote_device_nodemap.FindNode("TriggerMode").SetCurrentEntry("On")

		self.datastream = self.device.DataStreams()[0].OpenDataStream()
		payload_size = self.remote_device_nodemap.FindNode("PayloadSize").Value()
		for _ in range(self.datastream.NumBuffersAnnouncedMinRequired()):
			buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
			self.datastream.QueueBuffer(buffer)

		self.datastream.StartAcquisition()
		self.remote_device_nodemap.FindNode("AcquisitionStart").Execute()
		self.remote_device_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

	def stop(self):
		pass

	def read(self):
		
		# Trigger image
		self.remote_device_nodemap.FindNode("TriggerSoftware").Execute()
		buffer = self.datastream.WaitForFinishedBuffer(1000)
		raw_image = ids_ipl.Image_CreateFromSizeAndBuffer(buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), buffer.Width(), buffer.Height())

		# Convert to opencv image
		color_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_RGB8)
		self.datastream.QueueBuffer(buffer)
		image = color_image.get_numpy_3D()
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# Crop to lower resolution
		image = image[int((image.shape[0]-1080)/2):image.shape[0] - int((image.shape[0]-1080)/2), int((image.shape[1]-1920)/2):image.shape[1] - int((image.shape[1]-1920)/2)]
		return image, image
			


	


	
