# Imports
import cv2
import time
import yaml
import paho.mqtt.client as mqtt
from vision_demonstrator.Camera import Camera

# Load params
with open("config/demo4_config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Create camera object
cam = Camera("RealSense", config['color_resolution'], config['depth_resolution'], config['frames_per_second'], config['id'])

# Init MQTT server
client = mqtt.Client()
client.connect("mqtt.eclipseprojects.io")

# Load model params
model = models.resnet34()  # change to other models
for param in model.parameters():  # freezing all the parameters except the ones which are created after this for loop
    param.requires_grad = False

num_ftrs = model.fc.in_features

# Create model
model.fc = nn.Linear(num_ftrs, 3)  # the outputs are the amount of class labels
model.load_state_dict(torch.load('model2.pth'))
model.eval()

# Define classes
klassen = ["Large scratch", "No defect", "Nut"]
transform = transforms.ToTensor()

# Loop
while True:

    # Get start time
    t1 = time.time()

    # Read frame
    color_image, depth_image = cam.read()

    # Show image
    color = color[0:1080, 840:1920]
    tensor = transform(color)
    tensor = tensor.unsqueeze(0)
    output = model(tensor)
    cv2.putText(color, klassen[output.argmax()], [12, 1000], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
    
    ### End of loop

    # Write as image
    # cv2.imwrite('webserver/tmp/image4.jpg', color_image)

    # Publish data
    data = cv2.imencode('.jpg', color)[1].tobytes()
    client.publish("demo4_image", data)

    # Get end time
    t2 = time.time()

    # Sleep
    if (t2-t1) < 0.5: time.sleep(0.5 - (t2-t1))

    # Get end time
    t3 = time.time()

    # Print
    print("Demo 4 - classification - running at cycle time of " + str(t3-t1) + " seconds")


