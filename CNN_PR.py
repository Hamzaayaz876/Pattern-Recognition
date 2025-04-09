################################################
# 1. Imports
################################################

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product

################################################
# 2. Hardware
################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

################################################
# 3. Transformlar
################################################

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

################################################
# 4. Dataset & DataLoader
################################################

train_dir = "MNIST/train"
test_dir = "MNIST/test"

full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

################################################
# 5. Model Definition
################################################

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes=10, num_layers=2, kernel_size=3):
        super(FlexibleCNN, self).__init__()
        layers = []
        in_channels = 1
        out_channels = 32

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels * (28 // (2 ** num_layers)) ** 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

################################################
# 6. Training Function for Single Combination
################################################

def train_single_combination(kernel_size, num_layers, lr, num_epochs=5):
    model = FlexibleCNN(num_classes=10, num_layers=num_layers, kernel_size=kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        correct = total = 0
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)
        train_accs.append(correct / total)
        train_losses.append(running_loss / total)

        # Validation
        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item() * images.size(0)
        val_accs.append(correct / total)
        val_losses.append(val_loss / total)

    # Test accuracy hesapla
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)

    return train_accs, val_accs, train_losses, val_losses, test_acc

################################################
# 7. Hyperparameter Combinations
################################################

kernel_sizes = [3, 5]
num_layers_list = [1, 2]
learning_rates = [0.001, 0.01]
num_epochs = 5

param_grid = list(product(kernel_sizes, num_layers_list, learning_rates))
results = []

for k_size, n_layer, lr in param_grid:
    print(f"\n Training with kernel={k_size}, layers={n_layer}, lr={lr}")
    train_accs, val_accs, train_losses, val_losses, test_acc = train_single_combination(k_size, n_layer, lr, num_epochs)
    best_val_acc = max(val_accs)
    results.append({
        "kernel_size": k_size,
        "num_layers": n_layer,
        "lr": lr,
        "best_val_acc": best_val_acc,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_acc": test_acc
    })

################################################
# 8. Results Table
################################################

df_results = pd.DataFrame([{
    "Kernel Size": r["kernel_size"],
    "Num Layers": r["num_layers"],
    "Learning Rate": r["lr"],
    "Best Val Accuracy": round(r["best_val_acc"], 4),
    "Test Accuracy": round(r["test_acc"], 4)
} for r in results])

print("\n Grid Search Results (Sorted by Val Acc):")
print(df_results.sort_values(by="Best Val Accuracy", ascending=False))

################################################
# 9. Grafik: Accuracy & Loss – Training & Validation + Test
################################################

for r in results:
    label = f"k={r['kernel_size']}, l={r['num_layers']}, lr={r['lr']}"
    
    # Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r["train_accs"], label="Train Acc")
    plt.plot(r["val_accs"], label="Val Acc")
    plt.axhline(y=r["test_acc"], color='r', linestyle='--', label=f"Test Acc: {r['test_acc']:.2f}")
    plt.title(f"Accuracy - {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(r["train_losses"], label="Train Loss")
    plt.plot(r["val_losses"], label="Val Loss")
    plt.title(f"Loss - {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
