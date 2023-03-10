# Before running the script. Connect the camera via the USB3.0 cable to a USB3.0 port on your computer.
# Download pyrealsense2: pip install pyrealsense2

# Imports
import os
import cv2
import numpy
import pyrealsense2 as realsense
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

# Camera setup params
color_resolution = (1920, 1080)
depth_resolution = (1280, 720)
frames_per_second = 30
camera_id = '821312060313'  # To be found in Realsense Viewer

model = models.resnet34()  # change to other models
for param in model.parameters():  # freezing all the parameters except the ones which are created after this for loop
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 3)  # the outputs are the amount of class labels
model.load_state_dict(torch.load('model2.pth'))
model.eval()

klassen = ["Large scratch", "No defect", "Nut"]
transform = transforms.ToTensor()

# Connect camera
conn = realsense.pipeline()

# Config camera
conf = realsense.config()
conf.enable_device(camera_id)
conf.enable_stream(realsense.stream.depth, depth_resolution[0], depth_resolution[1], realsense.format.z16,
                   frames_per_second)
conf.enable_stream(realsense.stream.color, color_resolution[0], color_resolution[1], realsense.format.bgr8,
                   frames_per_second)

# Start streaming
conn.start(conf)

# Align images
align = realsense.align(realsense.stream.color)

while True:
    # Wait for image
    frames = conn.wait_for_frames()

    # Align images
    aligned_frames = align.process(frames)

    # Retrieve images
    color_frame = aligned_frames.get_color_frame()

    # Convert to arrays
    color = numpy.asanyarray(color_frame.get_data())

    # Show image
    cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)
    color = color[0:1080, 840:1920]
    tensor = transform(color)
    tensor = tensor.unsqueeze(0)
    output = model(tensor)
    cv2.imshow('Inference', cv2.putText(color, klassen[output.argmax()], [12, 1000], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3))

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC to exit
        print("Closing...")
        break

    """
    k = cv2.waitKey(1)

    if k % 256 == 32:
        # Inference
        transform = transforms.ToTensor()
        tensor = transform(color)
        tensor = tensor.unsqueeze(0)
        output = model(tensor)
        print(output)
        print(klassen[output.argmax()])
        cv2.imshow('Inference', cv2.putText(color, klassen[output.argmax()], [12, 500], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3))
    elif k % 256 == 27:  # ESC to exit
        print("Closing...")
        break
    
    """

cv2.destroyAllWindows()
