# Imports
import cv2
import numpy as np
import pyrealsense2 as rs

# Params
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

# Colors
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# Plot bounding boxes
def wrap_detection(input_image, output_data):

	# Init
	class_ids = []
	confidences = []
	boxes = []

	# Predictions
	rows = output_data.shape[0]

	# Params
	image_width, image_height, _ = input_image.shape
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	# Loop over predictions
	for r in range(rows):
		row = output_data[r]
		confidence = row[4]
		if confidence >= 0.4:
			classes_scores = row[5:]
			_, _, _, max_indx = cv2.minMaxLoc(classes_scores)
			class_id = max_indx[1]
			if (classes_scores[class_id] > .25):
				confidences.append(confidence)
				class_ids.append(class_id)
				x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
				left = int((x - 0.5 * w) * x_factor)
				top = int((y - 0.5 * h) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
				box = np.array([left, top, width, height])
				boxes.append(box)


	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

	result_class_ids = []
	result_confidences = []
	result_boxes = []

	for i in indexes:
		result_confidences.append(confidences[i])
		result_class_ids.append(class_ids[i])
		result_boxes.append(boxes[i])

	return result_class_ids, result_confidences, result_boxes

# Loop
def main():

	# Loop
	while True:

		# Get RGB frame from camera
		frames = pipeline.wait_for_frames()
		frame = frames.get_color_frame()
		frame = np.asanyarray(frame.get_data())
			
		# Resize to 640x640, normalize to [0,1] and swap Red and Blue channels
		col, row, _ = frame.shape
		_max = max(col, row)
		resized = np.zeros((_max, _max, 3), np.uint8)
		resized[0:col, 0:row] = frame
		blob = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)

		# Predict
		net.setInput(blob)
		preds = net.forward()

		# Draw detection
		class_ids, confidences, boxes = wrap_detection(resized, preds[0])
		for (classid, confidence, box) in zip(class_ids, confidences, boxes):
			color = colors[int(classid) % len(colors)]
			cv2.rectangle(frame, box, color, 2)
			cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
			cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

		# Resize image
		screensize = 4096, 2160
		resized = cv2.resize(frame, screensize, interpolation=cv2.INTER_AREA)

		# Show
		cv2.imshow("People detection", frame)
		if cv2.waitKey(1) > -1:
			break


if __name__ == '__main__':

	# Camera params
	color_resolution = (1920, 1080)
	depth_resolution = (1280, 720)
	frames_per_second = 30

	# Connect to realsense
	pipeline = rs.pipeline()

	# Config camera
	config = rs.config()
	config.enable_device('821212060746') #'821312060313'
	config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frames_per_second)
	config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, frames_per_second)
	
	# Start streaming
	pipeline.start(config)

	# Get model
	net = cv2.dnn.readNet("data/yolov5s.onnx")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	# Get classes list
	class_list = []
	with open("data/classes.txt", "r") as f:
		class_list = [cname.strip() for cname in f.readlines()]
		
	# Loop
	#try:
	main()
	#except:
		#pipeline.stop()
		#cv2.destroyAllWindows()