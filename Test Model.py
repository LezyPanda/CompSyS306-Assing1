import torch
import PIL
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.classifier(x)
        return x

# Load data
data = pd.read_csv('traffic signs class/labels.csv')

x, y = [], []  # X to store images and y to store respective labels  
data_dir = 'traffic signs class/myData'
for folder in range(43):
    folder_path = os.path.join(data_dir, str(folder))
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        img_tensor = np.array(PIL.Image.open(img_path))
        x.append(img_tensor)
        y.append(folder)
    print('folder of label', folder, 'images loaded. Number of samples:', len(os.listdir(folder_path)))

x = np.array(x)
y = np.array(y)

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=40),
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split dataset into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y)

# Convert labels to torch.LongTensor
ytrain = torch.LongTensor(ytrain)
ytest = torch.LongTensor(ytest)

# Create datasets
train_dataset = TrafficSignsDataset(xtrain, ytrain, transform=transform)
test_dataset = TrafficSignsDataset(xtest, ytest, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define model
model = AlexNet(num_classes=43).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler(device='cuda')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Train and test model
num_epochs = 30
train_losses, test_losses, test_accuracies = [], [], []
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total

    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    end_time = time.time()
    time_taken = end_time - start_time

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {time_taken:.2f}s')

# Plot the training loss, test loss, and test accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.show()