import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import customModel
from utils import *
import matplotlib.pyplot as plt 
import numpy as np

# Getting intermediate results
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = customModel.CNN()
model.load_state_dict(torch.load('PATH_TO_MODEL_WEIGHTS')) # update path here
model.to(device)
model.eval()

# Transform the data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 4 channels
# Get the test data
img_path="PATH_TO_IMG" # update the image here
img = [Image.open(img_path).convert('RGB')]
label= [0]
oneSet = CustomDataset(img,label,transform)
oneSet_loader = DataLoader(dataset=oneSet, batch_size=1, shuffle=False)

# Pass data through model 
with torch.no_grad():
  for img1,label1 in oneSet_loader:
    original=img1
    img1 = img1.to(device)
    a,b,c,d = model.getLayer1(img1) # get 4 channels
    output=model(img1)
    print(output.cpu())

original = torch.squeeze(original) #Get rid of batch dimension
original = original.cpu() # H,W

fig, axs = plt.subplots(1, 5)
# Plot each subplot with images
axs[0].imshow(original.permute(1,2,0),vmin=0,vmax=1)
axs[0].set_title('Input')
axs[1].imshow(a,cmap='gray',vmin=0,vmax=1)
axs[1].set_title('Channel 1')
axs[2].imshow(b,cmap='gray',vmin=0,vmax=1)
axs[2].set_title('Channel 2')
axs[3].imshow(c,cmap='gray',vmin=0,vmax=1)
axs[3].set_title('Channel 3')
axs[4].imshow(d,cmap='gray',vmin=0,vmax=1)
axs[4].set_title('Channel 4')
for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()

# 16 channels
# Get the test data
img_path="PATH_TO_IMG" # update the image here
img = [Image.open(img_path).convert('RGB')]
label= [0]
oneSet = CustomDataset(img,label,transform)
oneSet_loader = DataLoader(dataset=oneSet, batch_size=1, shuffle=False)
with torch.no_grad():
  for img1,label1 in oneSet_loader:
    original=img1
    img1 = img1.to(device)
    a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,d1,d2,d3,d4 = model.getLayer2(img1) # Get 16 channels

original = torch.squeeze(original) #Get rid of batch dimension
original = original.cpu() # H,W

fig, axs = plt.subplots(4, 4)
# Plot each subplot with images
axs[0,0].imshow(a1,cmap='gray',vmin=0,vmax=1)
axs[0,0].set_title('Channel 1')
axs[0,1].imshow(a2,cmap='gray',vmin=0,vmax=1)
axs[0,1].set_title('Channel 2')
axs[0,2].imshow(a3,cmap='gray',vmin=0,vmax=1)
axs[0,2].set_title('Channel 3')
axs[0,3].imshow(a4,cmap='gray',vmin=0,vmax=1)
axs[0,3].set_title('Channel 4')

axs[1,0].imshow(b1,cmap='gray',vmin=0,vmax=1)
axs[1,0].set_title('Channel 5')
axs[1,1].imshow(b2,cmap='gray',vmin=0,vmax=1)
axs[1,1].set_title('Channel 6')
axs[1,2].imshow(b3,cmap='gray',vmin=0,vmax=1)
axs[1,2].set_title('Channel 7')
axs[1,3].imshow(b4,cmap='gray',vmin=0,vmax=1)
axs[1,3].set_title('Channel 8')

axs[2,0].imshow(c1,cmap='gray',vmin=0,vmax=1)
axs[2,0].set_title('Channel 9')
axs[2,1].imshow(c2,cmap='gray',vmin=0,vmax=1)
axs[2,1].set_title('Channel 10')
axs[2,2].imshow(c3,cmap='gray',vmin=0,vmax=1)
axs[2,2].set_title('Channel 11')
axs[2,3].imshow(c4,cmap='gray',vmin=0,vmax=1)
axs[2,3].set_title('Channel 12')

axs[3,0].imshow(d1,cmap='gray',vmin=0,vmax=1)
axs[3,0].set_title('Channel 13')
axs[3,1].imshow(d2,cmap='gray',vmin=0,vmax=1)
axs[3,1].set_title('Channel 14')
axs[3,2].imshow(d3,cmap='gray',vmin=0,vmax=1)
axs[3,2].set_title('Channel 15')
axs[3,3].imshow(d4,cmap='gray',vmin=0,vmax=1)
axs[3,3].set_title('Channel 16')
for ax in axs:
  for subAx in ax:
      subAx.axis('off')

plt.tight_layout()
plt.show()
