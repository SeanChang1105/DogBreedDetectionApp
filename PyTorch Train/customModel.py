import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

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

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),

            nn.ReLU(),
            nn.BatchNorm2d(128),
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

    # Getting intermediate results (first 4 channels)
    def getLayer1(self,x):
        x = self.layer1(x)
        print(x.shape)
        x = torch.squeeze(x) #Get rid of batch dimension
        x = x.permute(1,2,0).cpu() # H,W,C
        return x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3]
    
    # Getting intermediate results (first 16 channels)
    def getLayer2(self,x):
        x = self.layer1(x)
        print(x.shape)
        x = torch.squeeze(x) #Get rid of batch dimension
        x = x.permute(1,2,0).cpu() # H,W,C
        return x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3],x[:,:,4],x[:,:,5],x[:,:,6],x[:,:,7],x[:,:,8],x[:,:,9],x[:,:,10],x[:,:,11],x[:,:,12],x[:,:,13],x[:,:,14],x[:,:,15]