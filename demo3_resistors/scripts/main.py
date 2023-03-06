# Imports
import cv2
import time
import pickle
import pandas as pd
from pypylon import pylon
import matplotlib.pyplot as plt
from decode import *
from preprocessing import *
from sklearn.preprocessing import LabelEncoder 

# Get camera instance
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Start grabbing
camera.Open()
camera.StartGrabbing(1)

# Converter for opencv
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Load data
df = pd.read_csv("demo3_resistors/data/color_data.csv")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'demo3_resistors/data/model.sav'
model = pickle.load(open(filename, 'rb'))

# Loop
while camera.IsGrabbing():

    # Get start time
    t1 = time.time()

    # Get frame
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab.GrabSucceeded():

        # Convert image
        image = converter.Convert(grab)
        image = image.GetArray()     

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract resistor
        ret, cropped = extract_resistor(image)

        # Check validity
        if ret: 

            # Extract color bands
            ret, color_bands = extract_color_bands(cropped)

            # Stop if the result is wrong
            if ret:

                # Iterate over first three contours
                prediction = ''
                for j, ctr in enumerate(color_bands):

                    # Get roi
                    x,y,w,h = cv2.boundingRect(ctr)
                    roi = cropped[y:y+h, x+5:x+w-5]

                    # Make training data
                    new_data = np.reshape(roi, (roi.shape[0]*roi.shape[1], roi.shape[2]))

                    # Continue if data has no Nan values
                    if not np.isnan(new_data).any():
                    
                        # Predict
                        pred = model.predict([[np.mean(new_data[:,0]), np.mean(new_data[:,1]), np.mean(new_data[:,2])]])

                        # Convert to class
                        pred = labelencoder.inverse_transform(pred)[0]
                        prediction += pred

                # Draw text
                if len(prediction) == 3:
                    cv2.putText(img=image, text=prediction + " - " + decode(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

        # Write as image
        cv2.imwrite('webserver/tmp/image3.jpg', image)

    # Print
    print("Demo 3 - resistors - running")

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))

    # Release grab
    grab.Release()
    
# Releasing the resource    
camera.StopGrabbing()
