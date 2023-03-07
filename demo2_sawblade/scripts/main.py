# Imports
import os
import cv2
import time
import numpy
import config
from ftplib import FTP
import pyrealsense2 as realsense

# File name
filematch = '*.jpg'
target_dir = "images"

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

    # Loop over the directories
    for directory in directories:
        ftp.cwd(parent +'/'+ directory)
        for image in ftp.nlst():
            local_filename = os.path.join('images', image)
            file = open(local_filename, 'wb')
            ftp.retrbinary('RETR '+ image, file.write)
            file.close()
            ftp.delete(image)
        try:
            ftp.rmd(parent +'/'+ directory)
        except:
            pass

    # Quit file transfer
    ftp.quit() 

    # Write as image
    cv2.imwrite('webserver/tmp/image2.jpg', file)

    # Print
    print("Demo 2 - sawblade - running")

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))