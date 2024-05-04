import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import customModel
from utils import *


# Load images from folder
data_folder="" #Update folder path here
trainimages,trainlabels=load_images_from_folder(data_folder)

# Ensure GPU integration
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 30
learning_rate = 0.001
num_epochs = 50

# Load and preprocess data
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(trainimages,trainlabels,transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = customModel.CNN().to(device)
criterion = nn.CrossEntropyLoss() # use cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
losses=[]
minLost=100 # documenting minimal loss
# Training loop
for epoch in range(num_epochs):
    epoch_loss=0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move to GPU
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    # Calculate average epoch loss
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)

    # Print intermediate results
    print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')
    if epoch>0.8*num_epochs and epoch_loss<minLost:# Save the model and update minimal loss
      minLost=epoch_loss
      torch.save(model.state_dict(), 'model_weights.pth') 
      print("Saved model in epoch ",epoch)

print("Training Finish")