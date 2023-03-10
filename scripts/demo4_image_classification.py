# Imports
import os
import cv2
import numpy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

image_path = "processed_data_3/no_defect/sample_50.jpg"

class_names = ["Large scratch", "No defect", "Nut"]

# Setup model
model = models.resnet34()
for param in model.parameters():  # freezing all the parameters except the ones which are created after this for loop
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # add layer
model.load_state_dict(torch.load('model6.pth'))  # load model
model.eval()


picture = Image.open(image_path)
transform = transforms.ToTensor()
tensor = transform(picture)  # Transform to tensor
tensor = tensor.unsqueeze(0)  # Add dimension
output = model(tensor)  # Inference
print(output)
print(output.argmax())
print(class_names[output.argmax()])

picture_text = cv2.putText(cv2.imread(image_path), class_names[output.argmax()], [12, 1000], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)

while True:
    cv2.imshow('Inference', picture_text)
    k = cv2.waitKey(1)
    if k % 256 == 32:
        break
