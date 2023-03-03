# Imports
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

# Load image
image = cv.imread('demo4_classification/data/validation/images/crazing/crazing_241')

# Define transforms for validation data
transform_test = transforms.Compose([
    transforms.Resize((150, 150)), # Becasue vgg takes 150*150
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# Get model
model = torch.load("modelvgg_intel.pth")
model.eval() # To disable dropout and regularization

# Transform image
input = trans(image)
input = input.view(1, 3, 32,32)

# Predict
output = model(input)
prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

# Interpret prediction
if (prediction == 0):
    print ('daisy')
if (prediction == 1):
    print ('dandelion')
if (prediction == 2):
    print ('rose')
if (prediction == 3):
    print ('sunflower')
if (prediction == 4):
    print ('tulip')

# Reshape image
image = image.reshape(28, 28, 1)
    
# Show result
plt.imshow(image, cmap='gray')
plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
plt.show()

