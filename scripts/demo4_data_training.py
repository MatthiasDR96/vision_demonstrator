import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import splitfolders
import matplotlib.pyplot as plt
from torchvision.models import ResNet34_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using {device} device")

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# transforms.compose is used to perform multiple sequential transformations on an image
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}
# SPLIT FOLDER INTO TRAIN AND VAL
data_dir = 'processed_data_3'  # change the path
sets = ['train', 'val']
# splitfolders.ratio(data_dir, output="split_data2", seed=1337, ratio=(.7, 0.3))

data_dir = "split_data2"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}  # a dict is created with key values train and val. The values are the transformed data from the respective folders.
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=0) for x in sets}  # random batches of size 10 are loaded instead of the whole dataset.
dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes

print(dataset_sizes, class_names)


def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


inputs, classes = next(iter(dataloaders['train']))  # iterate through the batch to obtain the individual images and its labels

out = torchvision.utils.make_grid(inputs, nrow=5)  # organizing the images in a grid structure

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(
        model.state_dict())  # takes a copy of the model and also of all the inner layers, weights, biases
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # store the inputs to the selected device
                labels = labels.to(device)  # store the labels to the selected device

                # forward step
                with torch.set_grad_enabled(
                        phase == 'train'):  # gradient calculation will only be enabled when in training mode
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward step and optimise only in train phase
                    if phase == 'train':
                        optimizer.zero_grad()  # Pytorch keeps track of the gradients over multiple epochs, therefore initialise it to 0
                        loss.backward()
                        optimizer.step()  # update the model parameters using the defined optimizer

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # change to other models

for param in model.parameters():  # freezing all the parameters except the ones which are created after this for loop
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, len(class_names))  # the outputs are the amount of class labels
# model.load_state_dict(torch.load('model4.pth'))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# StepLR decays the learning rate every 7 epochs by a factor of 0.1
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=30)

torch.save(model.state_dict(), 'model4.pth')
