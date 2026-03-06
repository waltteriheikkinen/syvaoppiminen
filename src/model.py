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
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset
from torch.utils.data import DataLoader

class ImprovedCNN(nn.Module):
    """
    Syvempi CNN vedenalaiskuville.
    4 konvoluutiokerrosta, batch normalization ja dropout.
    Binary classification (1 output).
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # 1. konvoluutio + BN + ReLU + Pool + Dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # 3. ja 4. konvoluutio + BN + ReLU + Pool + Dropout
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # FC-kerrokset
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)  # BCEWithLogitsLoss output

        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Convolutional block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        # Output
        x = self.fc_out(x)
        return x

# =========================
# Get model + loss + optimizer
# =========================
def get_model(device="cpu", lr=1e-3, pos_weight=199.0, num_classes=1):
    model = ImprovedCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

# =========================
# WeightedRandomSampler dataloaderille
# =========================
def get_train_loader(train_dataset, batch_size=32):
    # Kerätään labelit listaksi
    labels = [int(label) for _, label in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        sampler=sampler, num_workers=4)
    return loader

# =========================
# Train-epoch
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", ncols=100):
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# =========================
# Validation
# =========================
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", ncols=100):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return val_loss, val_acc, f1

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128 if DEVICE.type == "cuda" else 32
    print(f"Using device: {DEVICE}")
    
    # =========================
    # CUDA info
    # =========================
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    
    
    data_dir = Path("data/RODI-DATA/RODI-DATA/Train")
    
    # =========================
    # Hae alkuperäiset dataloaderit
    # =========================
    train_loader_orig, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    train_dataset = train_loader_orig.dataset  # dataset samplerille
    
    # =========================
    # WeightedRandomSampler train datasetille
    # =========================
    # Alkuperäinen dataset
    full_dataset = train_loader_orig.dataset  # ei Subset
    
    
    # Hae labels listaksi ilman kuvien latausta
    print("Lasketaan painoja")
    labels = [int(label) for _, label in full_dataset]  

    # Laske painot
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # =========================
    # Luo train_loader samplerilla
    # =========================
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )
    
    print("Dataloaders ready")

    # =========================
    # Malli ja optimizer
    # =========================
    model, criterion, optimizer = get_model(device=DEVICE, num_classes=1, lr=1e-3)
    print("Model ready")

       
    NUM_EPOCHS = 50
    patience = 5
    best_val_f1 = 0.0
    epochs_no_improve = 0

    f1_scores = []  # Tallennetaan jokaisen epokin F1

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)

        f1_scores.append(val_f1)  # Lisää F1 listaan

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Tallennetaan paras malli F1:n perusteella
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("  --> Parannus, tallennetaan malli")
        else:
            epochs_no_improve += 1
            print(f"  --> Ei parannusta ({epochs_no_improve}/{patience})")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: F1-score ei parantunut {patience} epokkiin")
            break

    print(f"\nTraining finished.")
    print(f"Best validation F1-score: {best_val_f1:.4f}")

    # =====================
    # Piirretään F1-kaavio
    # =====================
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='o', color='b')
    plt.title("Validation F1-score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()