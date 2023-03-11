import os
import copy
import torch
import numpy as np
import splitfolders
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet34_Weights
from torchvision import datasets, models, transforms

# Define normalisation params
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Transforms.compose is used to perform multiple sequential transformations on an image
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Split data
#splitfolders.ratio("data/defect_images", output="data/defect_images", seed=1337, ratio=(0.7,0.3))

# Split data in train and test
data_dir = "data/defect_images"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}  # a dict is created with key values train and val. The values are the transformed data from the respective folders.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=0) for x in ['train', 'val']}  # random batches of size 10 are loaded instead of the whole dataset.
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Train model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    # Take a copy of the model and also of all the inner layers, weights, biases
    best_model_wts = copy.deepcopy(model.state_dict())  
    best_acc = 0.0

    # Loop over all epochs
    for epoch in range(num_epochs):

        # Print
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Loop over training and validation phase
        for phase in ['train', 'val']:

            # Set model state
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:

                # Set to device
                inputs = inputs.to(device)  # store the inputs to the selected device
                labels = labels.to(device)  # store the labels to the selected device

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward step
                with torch.set_grad_enabled(phase == 'train'):  # gradient calculation will only be enabled when in training mode

                    # Predict
                    outputs = model(inputs)

                    # Get predicted lable
                    _, preds = torch.max(outputs, 1)

                    # Calculate loss
                    loss = criterion(outputs, labels)

                    # Backward step and optimise only in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Debug
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step scheduler
            if phase == 'train':
                scheduler.step()

            # Calculate losses
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Print
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Print
    print('Best val Acc: {:.4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model

# Get trained model
model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features

# Freeze all the parameters except the ones which are created after this for loop
for param in model.parameters(): param.requires_grad = False

# Create new classification model
model.fc = nn.Linear(num_ftrs, len(class_names))  

# Link model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model.to(device)

# Set learning params
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# StepLR decays the learning rate every 7 epochs by a factor of 0.1
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Train model
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=30)

# Save model
torch.save(model.state_dict(), 'data/model4.pth')
