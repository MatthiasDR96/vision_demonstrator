# Imports
import os
import cv2
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decode import decode
from sklearn.preprocessing import LabelEncoder 
from preprocessing import *

# Load data
df = pd.read_csv("demo3_resistors/data/color_data.csv")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'demo3_resistors/data/model.sav'
model = pickle.load(open(filename, 'rb'))

# Loop over every image
for i in glob.glob('demo3_resistors/data/images/*jpg'):

    # Read image
    image = cv2.imread(i)

    # Read label
    label = i.split('/')[-1].split('_')[1][0:-4]

    # Extract resistor
    cropped = extract_resistor(image)

    # Extract color bands
    color_bands = extract_color_bands(cropped)

    # Stop if the result is wrong
    if color_bands is None:
        continue

    # Iterate over first three contours
    prediction = ''
    for j, ctr in enumerate(color_bands):

        # Get roi
        x,y,w,h = cv2.boundingRect(ctr)
        roi = cropped[y:y+h, x+5:x+w-5]

        # Make training data
        new_data = np.reshape(roi, (roi.shape[0]*roi.shape[1], roi.shape[2]))

        # Predict
        pred = model.predict([[np.mean(new_data[:,0]), np.mean(new_data[:,1]), np.mean(new_data[:,2])]])

        # Convert to class
        pred = labelencoder.inverse_transform(pred)[0]
        prediction += pred

    # Print result
    print(decode(label) + ' - ' + decode(prediction))

    # Plot
    cv2.putText(img=image, text=decode(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
    plt.imshow(image)
    plt.show()

