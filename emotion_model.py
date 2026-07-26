import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader   # Loads data in batches (e.g., 25 images at a time) and can shuffle the data.
from torchvision import datasets, transforms
from PIL import Image     #Used to open images during prediction.
import numpy as np
import os




# -----------------------------------------------
# Image Transform
# ------------------------------------------------


transform = transforms.Compose([
     
    transforms.Resize((48,48)),## Ater resizing image size is (200 × 200 × 3)
    transforms.ToTensor(),        # ToTensor() converts the image into a 3D tensor (3, H, W) and scales pixel values from 0–255 to 0–1.
    transforms.Normalize(         ## Each RGB channel is normalized using mean = 0.5 and std = 0.5.
        mean=(0.5,0.5,0.5),
        std=(0.5,0.5,0.5)
    )
])


train_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )])
# -----------------------------
# Dataset
# -----------------------------



train_dataset = datasets.ImageFolder(                   #ImageFolder automatically reads images from folders and assigns labels.
    r"C:\Users\yashr\OneDrive\Desktop\Emotion_Detection\training",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    r"C:\Users\yashr\OneDrive\Desktop\Emotion_Detection\validation",
    transform=transform
)


train_loader = DataLoader(               ## DataLoader : define how the data will be loaded for traing 
    train_dataset,
    batch_size=25,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=25,
    shuffle=False
)

# -----------------------------
# CNN Model
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            

            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3 input channels (RGB), apply 16 learnable 3×3×3 filters, producing 16 output feature maps.

            nn.BatchNorm2d(16),                        # Normalizes the 16 feature maps, making training faster and more stable.

            nn.ReLU(),                                 # Replaces negative values with 0 in all output feature maps.

            nn.MaxPool2d(kernel_size=2, stride=2),  # Uses a 2×2 window, keeps the maximum value, reducing each feature map size by half.         




            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )

        self.fc = nn.Sequential(

            nn.Flatten(),             # 64×25×25 → 40000 features
            

            nn.Linear(64*6*6,512),    # 64×25×25 → 40000 features
            nn.ReLU(),

            nn.Linear(512,7)           # 512 input features → 7 output classes (emotions)  

                                       ## 1 array containing 7 values (one value from each neuron).
        )

    
    def forward(self,x):

        x = self.conv_layers(x)
        x = self.fc(x)

        return x


model = CNN()

# -----------------------------
# Loss & Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()      ## loss function

optimizer = optim.Adam(                ## optimizer
    model.parameters(),
    lr=0.001
)
# -----------------------------
# Training
# -----------------------------


epochs = 10

for epoch in range(epochs):              # Train the model 10 times on the entire dataset.

    model.train()                         # Set model to training mode.   

    running_loss = 0                       # Stores total loss for this epoch.

    for images, labels in train_loader:     # Get one batch of images and labels.

        optimizer.zero_grad()                # Clear old gradients.

        outputs = model(images)            

        loss = criterion(outputs, labels)    # Compute loss.

        loss.backward()                      # Backpropagation: compute gradients.

        optimizer.step()                      # Update model weights and bias.

        running_loss += loss.item()           # Add batch loss to total loss.

    print(f"Epoch {epoch+1} Loss : {running_loss/len(train_loader):.4f}")      # Average loss for the epoch.
 

# ----------------------------------
# Validation Accuracy
# -----------------------------------
model.eval()                              # Set model to evaluation mode.

correct = 0                               # Number of correct predictions.
total = 0                                 # Total number of validation images.

with torch.no_grad():   

    for images, labels in validation_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

print("Validation Accuracy :",100*correct/total)









# # -----------------------------------------------------------
# # Prediction
# # -----------------------------------------------------------
# class_names = train_dataset.classes

# test_folder = r"C:\Users\yashr\OneDrive\Desktop\cnn model\testing"

# model.eval()

# for file in os.listdir(test_folder):

#     img = Image.open(os.path.join(test_folder,file)).convert("RGB")

#     img = transform(img)

#     img = img.unsqueeze(0)

#     with torch.no_grad():

#         output = model(img)

#         prediction = torch.argmax(output,1).item()

#     print(file,"->",class_names[prediction])




# -----------------------------
# Save Model
# -----------------------------

torch.save(model.state_dict(), "mood_model.pth")

print("Model Saved")




