# Imports
import os
import cv2
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create empty dataset
df = pd.DataFrame({'R': pd.Series(dtype='int'),
                   'G': pd.Series(dtype='int'),
                   'B': pd.Series(dtype='int'),
                   'Class': pd.Series(dtype='str')})

# Loop over every image
for i in os.listdir('demo3_resistors/data'):

    # Read image
    image = cv2.imread('demo3_resistors/data/' + i)

    # Read label
    label = i.split('_')[1][0:-4]

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to gray, and threshold
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold background
    _, threshed = cv2.threshold(image_gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Morphological transformations to remove sticks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    morphed_open = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    morphed_close = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel)

    # Find contour of resistor
    maxcontour = max(cv2.findContours(morphed_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)

    # Get minimal area rectangle
    rect = cv2.minAreaRect(maxcontour)

    # Get rectangle properties
    angle = rect[2]
    rows, cols = image.shape[0], image.shape[1]

    # Rotate image
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle-90, 1)
    img_rot = cv2.warpAffine(image,M,(cols,rows))

    # Rotate bounding box 
    box = cv2.boxPoints((rect[0], rect[1], angle))
    pts = np.intp(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # Cropping
    cropped = img_rot[pts[0][1]+20:pts[3][1]-100, pts[0][0]+40:pts[2][0]-140]

    # Bilateral filtering
    cropped = cv2.bilateralFilter(cropped, 15, 35, 35)

    # Remove area in between color bands
    mask = cv2.bitwise_not(cv2.inRange(cropped, np.array([120, 120, 110]), np.array([190, 190, 180])))

    plt.figure(1)
    plt.imshow(cropped)
    plt.figure(2)
    plt.imshow(mask)
    plt.show()

    # Find the three largest contours of the color bands
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Check if there are only three contours
    if len(contours) < 3:
        continue

    # Get three largest contours
    largest_contours = sorted(contours, key=cv2.contourArea)
    largest_contours.reverse()
    largest_contours = largest_contours[0:3]

    # Plot
    tmp = cv2.drawContours(cropped.copy(), largest_contours, -1, (0,255,0), 1)
    plt.imshow(tmp)
    plt.show()

    # Sort contours from left to right
    sorted_contours = sorted(largest_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    sorted_contours.reverse()

    # Iterate over first three contours
    for j, ctr in enumerate(sorted_contours):

        # Plot
        #tmp = cv2.drawContours(cropped.copy(), ctr, -1, (0,255,0), 1)
        #plt.imshow(tmp)
        #plt.show()

        # Get roi
        x,y,w,h = cv2.boundingRect(ctr)
        roi = cropped[y:y+h, x+5:x+w-5]

        # Make training data
        new_data = np.reshape(roi, (roi.shape[0]*roi.shape[1], roi.shape[2]))
        labels = np.reshape(np.repeat(label[j], roi.shape[0]*roi.shape[1]), (roi.shape[0]*roi.shape[1], 1))

        # Make dataframe from sample
        df_tmp = pd.DataFrame()
        df_tmp['R'] = new_data[:, 0]
        df_tmp['G'] = new_data[:, 1]
        df_tmp['B'] = new_data[:, 2]
        df_tmp['Class'] = labels

        # Append to main dataframe
        df = pd.concat([df, df_tmp])

# Save data
df.to_csv('demo3_resistors/color_data', index=False)


