# Imports
import cv2
import time
import yaml
import pickle
import pandas as pd
import paho.mqtt.client as mqtt
from vision_demonstrator.decode import *
from vision_demonstrator.Camera import Camera
from vision_demonstrator.preprocessing import *
from sklearn.preprocessing import LabelEncoder 

# Load params
with open("config/demo3_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera('Basler', 0, 0, 0, 0)

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Load data
df = pd.read_csv("data/color_data.csv")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'data/model.sav'
model = pickle.load(open(filename, 'rb'))

# Loop
while True:

    # Get start time
    t1 = time.time()

    # Get frame
    image, _ = cam.read()

    # Check frame
    if image is None: continue

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
                
                    try:

                        # Predict
                        pred = model.predict([[np.mean(new_data[:,0]), np.mean(new_data[:,1]), np.mean(new_data[:,2])]])
    
                        # Convert to class
                        pred = labelencoder.inverse_transform(pred)[0]
                        prediction += pred

                    except:
                        pass

            # Draw text
            if len(prediction) == 3:
                cv2.putText(img=image, text=prediction + " - " + decode(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

    ### End of loop

    # Write as image
    # cv2.imwrite('webserver/tmp/image3.jpg', image)
    
    # Publish data
    data = cv2.imencode('.jpg', image)[1].tobytes()
    client.publish("demo3_image", data)

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.2: time.sleep(0.2 - (t2-t1))

    # Get end time
    t3 = time.time()

    # Print
    print("Demo 3 - resistors - running at cycle time of " + str(t3-t1) + " seconds")
