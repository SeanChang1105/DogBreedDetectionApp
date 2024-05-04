import os
from PIL import Image
import torch
from torch.utils.data import  Dataset

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_label, class_name in enumerate(os.listdir(folder)):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Convert to RGB format
            images.append(img)
            if class_name=="golden_retriever": #Labeling
              labels.append(0)
            else:
              labels.append(1)
    return images, labels

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