### https://www.pythonforthelab.com/blog/getting-started-with-basler-cameras/

# Imports
from pypylon import pylon
import cv2

# Check devices
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
for device in devices:
    print(device.GetFriendlyName())

# Get camera
tl_factory = pylon.TlFactory.GetInstance()
camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice())

# Get frame
camera.Open()
camera.StartGrabbing(1)

# Load data
df = pd.read_csv("demo3_resistors/color_data")

# Encode categorical labels
labelencoder = LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Load model
filename = 'demo3_resistors/model.sav'
model = pickle.load(open(filename, 'rb'))

# Loop
try:
    while true:

        # Grab frame
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():
            image = grab.GetArray()

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
            cropped = img_rot[pts[0][1]+20:pts[3][1]-100, pts[0][0]+40:pts[2][0]-40]

            # Bilateral filtering
            cropped = cv2.bilateralFilter(cropped, 15, 35, 35)

            # Remove area in between color bands
            mask = cv2.bitwise_not(cv2.inRange(cropped, np.array([120, 120, 100]), np.array([190, 180, 160])))

            # Find the contours of the color bands
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # Sort contours from left to right
            sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
            sorted_ctrs.reverse()

            # Iterate over first three contours
            prediction = ''
            for j, ctr in enumerate(sorted_ctrs[0:3]):

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

            # Plot
            cv2.putText(img=image, text=decode(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
            plt.imshow(image)
            plt.show()

except:
    camera.Close()