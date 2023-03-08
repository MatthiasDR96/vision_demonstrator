# Imports
import cv2
import glob
import pandas as pd
import numpy as np
from vision_demonstrator.preprocessing import *

# Create empty dataset
df = pd.DataFrame({'R': pd.Series(dtype='int'),
                   'G': pd.Series(dtype='int'),
                   'B': pd.Series(dtype='int'),
                   'Class': pd.Series(dtype='str')})

# Loop over every image
for i in glob.glob('data/images/*jpg'):
    
    # Read image
    image = cv2.imread(i)

    # Read label
    label = i.split('/')[-1].split('_')[1][0:-4]

    # Extract resistor
    cropped = extract_resistor(image)

    # Extract color bands
    color_bands = extract_color_bands(cropped)

    # Iterate over color bands
    for j, ctr in enumerate(color_bands):

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
df.to_csv('data/color_data.csv', index=False)


