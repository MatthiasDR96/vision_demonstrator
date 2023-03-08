# Imports
import cv2
import time
import matplotlib.pyplot as plt
from ftplib import FTP

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

        # Go to image folder
        ftp.cwd(parent +'/'+ directory)

        # Loop over images
        for image_path in ftp.nlst():

            # Write to file
            local_filename = 'webserver/tmp/image2.jpg'
            file = open(local_filename, 'wb')
            ftp.retrbinary('RETR '+ image_path, file.write)
            file.close()

            # Delete frame
            ftp.delete(image_path)

            # Delete directory
            ftp.rmd(parent +'/'+ directory) 

    # Quit ftp
    ftp.quit()  

    ### End of loop

    # Print
    print("Demo 2 - sawblade - running")

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))