import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from src.fish_data import get_dataloaders
from src.model import get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun


class SimpleCNN(nn.Module):
    """
    Perus 2-kerroksinen CNN vedenalaiskuville.
    Syötteenä 3-kanavainen RGB-kuva (esim. 50x50)
    Lopputuloksena 2-luokkaa (binary classification)
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Ensimmäinen konvoluutiokerros: 3-kanavaa -> 32 kanavaa
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Toinen konvoluutiokerros: 32 -> 64 kanavaa
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # FC-kerrokset
        # 2 poolauksen jälkeen 50x50 -> 25x25 -> 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Aktivointi ja dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv + ReLU + Pool
        x = self.pool(self.relu(self.conv1(x)))  # 50x50 -> 25x25
        x = self.pool(self.relu(self.conv2(x)))  # 25x25 -> 12x12

        # Flatten
        x = x.view(-1, 64 * 12 * 12)

        # Fully connected
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_model(device="cpu", lr=1e-3, num_classes=2):
    """
    Luodaan SimpleCNN, siirretään device:lle ja palautetaan optimizer + loss function.
    """
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\waltteri\projects\kurssit\syvaoppiminen\project_work\data\RODI-DATA\RODI-DATA\Train"
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=32)
    model, criterion, optimizer = get_model(device=DEVICE, num_classes=2, lr=1e-3)


    # Testataan batch
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        print("Images:", images.shape)
        print("Outputs:", outputs.shape)
        print("Labels:", labels.shape)
        break


if __name__ == "__main__":
    main()
