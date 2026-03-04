import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun

from src.fish_data import get_dataloaders


class SimpleCNN(nn.Module):
    """
    Perus 2-kerroksinen CNN vedenalaiskuville.
    Syötteenä 3-kanavainen RGB-kuva (esim. 50x50)
    Lopputuloksena 2-luokkaa (binary classification)
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_model(device="cpu", lr=1e-3, num_classes=2):
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([199.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # F1-laskentaa varten
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Binary F1-laskenta (positiivinen luokka = 1)
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return val_loss, val_acc, f1

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\Users\miika\Desktop\koulujutut\deeplearning\syvaoppiminen\data\RODI-DATA\RODI-DATA\Train"
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=32)
    print("dataloader doned")
    model, criterion, optimizer = get_model(device=DEVICE, num_classes=2, lr=1e-3)
    print("model lataus doned")

    # Testataan inputin toimivuus
    print("Testataan inputin toimivuus..")


    NUM_EPOCHS = 5  # Voit muuttaa

    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Tallennetaan paras malli F1:n perusteella
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\nTraining finished.")
    print(f"Best validation F1-score: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()