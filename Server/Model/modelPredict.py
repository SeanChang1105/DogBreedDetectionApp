import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import numpy as np
from PIL import Image

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target

#Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
            

        # input = channels * (height after conv/2) * (width after conv/2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust input size according to the output size of the last layer
        self.fc2 = nn.Linear(512, 2)  # Output layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    # transform data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # SPECIFY MODEL PATH AND TEST IMAGE PATH HERE
    # SPECIFY MODEL PATH AND TEST IMAGE PATH HERE

    modelPath="/data/lam138/Model/model_best_weights_2.pth"
    img_path="/data/lam138/DogImage/dog.jpg"

    # SPECIFY MODEL PATH AND TEST IMAGE PATH HERE
    # SPECIFY MODEL PATH AND TEST IMAGE PATH HERE

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = CNN()
    model.load_state_dict(torch.load(modelPath))
    model.to(device)
    model.eval()


    img = [Image.open(img_path).convert('RGB')]
    label= [0]
    oneSet = CustomDataset(img,label,transform)
    oneSet_loader = DataLoader(dataset=oneSet, batch_size=1, shuffle=False)
    with torch.no_grad():
        for img,_ in oneSet_loader:
            img = img.to(device)
            output = model(img)
            _, predicted = torch.max(output, 1)
            predictLabel=predicted.item()

    breed = "Golden Retriever" if predictLabel == 0 else "Siberian Husky"

    print(f"{breed}")