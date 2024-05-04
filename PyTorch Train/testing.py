import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import customModel
from utils import *
import matplotlib.pyplot as plt


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = customModel.CNN()
model.load_state_dict(torch.load('PATH_TO_MODEL_WEIGHTS')) # Update path here
model.to(device)

# Load and preprocess data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

data_folder="" #Update folder path here
testimages,testlabels=load_images_from_folder(data_folder)

test_dataset = CustomDataset(testimages,testlabels,transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

total=0
correct=0
model.eval()
wrongImgList=[]
wrongPredList=[]
idx=0

# Testing loop
for images,labels in test_loader:
  images, labels = images.to(device), labels.to(device)
  outputs = model(images)
  _, predicted = torch.max(outputs, 1) #class with maximum probability
  total += labels.size(0)
  correct += (predicted == labels).sum().item()

  if(predicted.item()!=labels.item()): # Get wrong predictions
    wrongImgList.append(images.cpu())
    wrongPredList.append(predicted.item())
    print(outputs.cpu())
    print("Wrong Id: ",idx)
  idx+=1

accuracy = correct / total
print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))


# Plot each subplot with wrong predictions
fig, axs = plt.subplots(1, len(wrongPredList))
for i in range(len(wrongPredList)):
  axs[i].imshow(torch.squeeze(wrongImgList[i]).permute(1,2,0),vmin=0,vmax=1)
  axs[i].set_title(wrongPredList[i])

for ax in axs:
    ax.axis('off')

plt.show()