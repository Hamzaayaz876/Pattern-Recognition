#*******************************************
#*
# Pattern Recogniton Group Asignemt S2025
#*********************************************
# MNIST classification using MLP

# 1st Step: Importing necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

# Defining dataset path
base_path = r'./MNIST-full'#MAKE SURE YOU ARE USING THE CORRECT MNIST DATASET PATH  
TRAIN_TSV = os.path.join(base_path, 'gt-train.tsv')
TEST_TSV = os.path.join(base_path, 'gt-test.tsv')
TRAIN_DIR = os.path.join(base_path, 'train')
TEST_DIR = os.path.join(base_path, 'test')

# MNIST Dataset Class
class MNISTDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, sep='\t', header=None)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        relative_img_path = self.img_labels.iloc[idx, 0].replace('/', os.sep)
        parts = relative_img_path.split(os.sep)
        if parts[0] in ['train', 'test']:
            relative_img_path = os.sep.join(parts[1:])

        img_path = os.path.join(self.img_dir, relative_img_path)
        image = read_image(img_path).float() / 255.0
        label = self.img_labels.iloc[idx, 1]
        return image.view(-1), label

# Load datasets
train_dataset = MNISTDataset(TRAIN_TSV, TRAIN_DIR)
test_dataset = MNISTDataset(TEST_TSV, TEST_DIR)

# Create train-validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(x)

model = MLP(hidden_size=128)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses, val_losses, train_acc, val_acc = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    total_train, correct_train, running_loss = 0, 0, 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_acc.append(correct_train / total_train)

    model.eval()
    total_val, correct_val, val_loss = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_acc.append(correct_val / total_val)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}')

# Ploting of training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate final model on test set
model.eval()
total_test, correct_test = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

print(f'Final Test Accuracy: {correct_test / total_test:.4f}')
