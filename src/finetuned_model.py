# src/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
import time
import json
import os
from pathlib import Path
import sys
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun
from src.data_finetuned import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
DATA_DIR = Path("../data/RODI-DATA/RODI-DATA/Train")


def compute_pos_weight(train_loader):
    """Laskee pos_weight harvalle luokalle BCEWithLogitsLossia varten"""
    num_pos = 0
    num_neg = 0
    for images, labels in train_loader:
        num_pos += labels.sum().item()
        num_neg += (labels == 0).sum().item()
    pos_weight = num_neg / max(num_pos, 1)
    return torch.tensor([pos_weight], device=DEVICE)


def get_model():
    """Luo pretrained ResNet18 ja muokkaa viimeisen fc-layerin binääriluokkaan"""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze kaikki kerrokset
    for param in model.parameters():
        param.requires_grad = False
    
    # Vaihda fc
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    # Unfreeze fc
    for param in model.fc.parameters():
        param.requires_grad = True

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze layer3
    for param in model.layer3.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):

    best_f1 = 0
    history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
    
        train_labels = []
        train_preds = []
    
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).float().to(DEVICE)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * images.size(0)
    
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())
    
        train_loss /= len(train_loader.dataset)
    
        train_acc = (np.array(train_preds) == np.array(train_labels)).mean()
    
        # Validointi
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []
    
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.unsqueeze(1).float().to(DEVICE)
    
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
    
                preds = (torch.sigmoid(outputs) > 0.5).long()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
    
        val_loss /= len(val_loader.dataset)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_f1 = f1_score(all_labels, all_preds)
    
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )
        
        # tallenna metrics historiaan
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        # tallenna paras malli
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "model_state_dict": model.state_dict()
            }, "best_finetuned_model.pt")
            print("Tallennettiin uusi paras malli")
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Aikaa epokkiin meni: {elapsed:.1f} sekuntia")

        with open("training_metrics_finetuned.json", "w") as f:
            json.dump(history, f, indent=4)

def main():
    start_time = time.time()
    print('Ajo alkoi...')
    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        image_size=IMAGE_SIZE
    )

    # Pos weight harvalle luokalle. Voi koittaa muutakin painotusta. Nyt laskettu on varmaan liian pieni
    pos_weight = compute_pos_weight(train_loader)
    #pos_weight = torch.tensor([10], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Malli ja optimizer
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Resnet18 ladattu: {elapsed:.1f} sekuntia\nMallin koulutus alkaa...")
    
    # Treenaa
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()