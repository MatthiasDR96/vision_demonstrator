# Imports
import cv2
import time
from ftplib import FTP
import paho.mqtt.client as mqtt

# Script rate
rate = 0.2 # Seconds per loop

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Loop
while True:

    # Get start time
    t1 = time.time()

    # Start file transfer
    ftp = FTP("10.5.5.100")
    ftp.login()
    ftp.cwd("RAMDisk")
    parent = ftp.pwd()

    # Get filenames within the directory
    directories = ftp.nlst()

    # Check if there is a directory
    if len(directories) == 0: continue

    # Go to most recent image folder
    ftp.cwd(parent +'/'+ directories[0])

    # Get images in folder
    image_paths = ftp.nlst()

    # Check if there is an image
    if len(image_paths) == 0: continue

    # Select required image
    image_path = image_paths[0]

    # Write to file
    file = open('webserver/tmp/image2.jpg', 'wb')
    ftp.retrbinary('RETR '+ image_path, file.write)
    file.close()

    # Delete frame
    ftp.delete(image_path)

    # Delete directory
    ftp.rmd(parent +'/'+ directories[0]) 

    # Publish data
    final_image = cv2.imread('webserver/tmp/image2.jpg')
    data = cv2.imencode('.jpg', final_image)[1].tobytes()
    client.publish("demo2_image", data)

    # Quit ftp
    ftp.quit()  

    ### End of loop

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < rate: time.sleep(rate - (t2-t1))

    # Get end time
    t3 = time.time()

    # Print
    print("Demo 2 - sawblade - running at cycle time of " + str(t3-t1) + " seconds")

