import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_random_weights import get_dataloaders
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler, Subset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

OUTPUT_DIR = Path("Pauluoutputs")

class ImprovedCNN(nn.Module):
    """
    Deeper CNN for underwater image classification.
    4 convolutional layers with batch normalization and dropout.
    Binary classification output.
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Block 1: Conv -> BN -> ReLU -> MaxPool -> Dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2: Conv -> BN -> ReLU -> MaxPool -> Dropout
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)  # Binary output for BCEWithLogitsLoss

        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass.
        Input: x (torch.Tensor) of shape [batch_size, 3, H, W]
        Output: torch.Tensor of shape [batch_size, 1]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        x = self.fc_out(x)
        return x


# =========================
# Model, loss, optimizer creation
# =========================
def get_model(device="cpu", lr=1e-3, pos_weight=10.0, num_classes=1):
    """
    Create model, loss function, and optimizer.
    Input:
        device: torch device ('cpu' or 'cuda')
        lr: learning rate
        pos_weight: weight for positive class in BCE loss
        num_classes: number of output classes (default 1)
    Output:
        model: torch.nn.Module
        criterion: loss function
        optimizer: optimizer instance
    """
    model = ImprovedCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(pos_weight)
    return model, criterion, optimizer


# =========================
# Create weighted train loader
# =========================
def get_train_loader(train_dataset, batch_size=32):
    """
    Create DataLoader with WeightedRandomSampler for class balancing.
    Input:
        train_dataset: PyTorch dataset
        batch_size: int
    Output:
        DataLoader with weighted sampling
    """
    labels = [int(label) for _, label in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    loader = DataLoader(train_dataset, batch_size=batch_size,
                        sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    return loader


# =========================
# Train one epoch function
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, device, debug=False):
    """
    Train the model for one epoch.
    Input:
        model: PyTorch model
        train_loader: DataLoader for training
        criterion: loss function
        optimizer: optimizer
        device: torch device
        debug: bool, print batch info if True
    Output:
        epoch_loss: float, average loss over epoch
        epoch_acc: float, accuracy over epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        if debug:
            num_fish = (labels == 1).sum().item()
            batch_size = labels.size(0)
            print(f"Batch {batch_idx+1}: {num_fish}/{batch_size} images are fish")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.7).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# Validation with detailed metrics
# =========================
def validate_with_metrics(model, val_loader, criterion, device, threshold=0.7):
    """
    Validate model and compute metrics.
    Input:
        model: PyTorch model
        val_loader: DataLoader for validation
        criterion: loss function
        device: torch device
        threshold: float, decision threshold for positive class
    Output:
        metrics: dict containing val_loss, accuracy, F1, precision, recall, AUROC, AUPRC
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", ncols=100):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            predicted = (probs > threshold).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    val_loss = running_loss / total
    accuracy = correct / total

    precision_fish = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_fish = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_fish = 2 * precision_fish * recall_fish / (precision_fish + recall_fish) if (precision_fish + recall_fish) > 0 else 0.0

    precision_nonfish = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_nonfish = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auroc = 0.0
        auprc = 0.0

    metrics = {
        "val_loss": val_loss,
        "accuracy": accuracy,
        "f1_fish": f1_fish,
        "precision_fish": precision_fish,
        "recall_fish": recall_fish,
        "precision_nonfish": precision_nonfish,
        "recall_nonfish": recall_nonfish,
        "auprc": auprc,
        "auroc": auroc
    }

    return metrics


# =========================
# Full training loop with early stopping and LR scheduler
# =========================
def run_training(model, train_loader, val_loader, criterion, optimizer, device,
                 num_epochs=50, patience=10, threshold=0.7, model_name="model_random_weights.pt"):
    """
    Train the model for multiple epochs with validation, early stopping, and LR scheduling.
    Input:
        model: PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        criterion: loss function
        optimizer: optimizer
        device: torch device
        num_epochs: int, maximum number of epochs
        patience: int, epochs to wait for improvement before stopping
        threshold: float, decision threshold for positive class
        model_name: str, path to save best model
    Output:
        f1_scores: list of F1 scores per epoch
        precision_scores: list of precision scores per epoch
        recall_scores: list of recall scores per epoch
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3,
        patience=3
    )

    best_val_f1 = 0.0
    epochs_no_improve = 0

    precision_scores = []
    recall_scores = []
    f1_scores = []

    history = []

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validate
        metrics = validate_with_metrics(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold
        )

        # Update learning rate based on F1 score
        scheduler.step(metrics["f1_fish"])

        # Save metrics
        f1_scores.append(metrics["f1_fish"])
        precision_scores.append(metrics["precision_fish"])
        recall_scores.append(metrics["recall_fish"])

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1 fish: {metrics['f1_fish']:.4f}"
        )
        print(
            f"Precision fish: {metrics['precision_fish']:.4f} | "
            f"Recall fish: {metrics['recall_fish']:.4f}"
        )
        print(
            f"Precision non-fish: {metrics['precision_nonfish']:.4f} | "
            f"Recall non-fish: {metrics['recall_nonfish']:.4f}"
        )
        print(
            f"AUPRC: {metrics['auprc']:.4f} | "
            f"AUROC: {metrics['auroc']:.4f}"
        )

        # Save best model
        if metrics["f1_fish"] > best_val_f1:
            best_val_f1 = metrics["f1_fish"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_f1": best_val_f1
            }, OUTPUT_DIR / model_name)
            print("  --> Improvement, saving model")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement ({epochs_no_improve}/{patience})")

        # tallenna metrics historiaan
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics['val_loss'],
            "val_acc": metrics['accuracy'],
            "val_f1": metrics['f1_fish'],
            "precision_fish": metrics['precision_fish'],
            "recall_fish": metrics['recall_fish'],
            "precision_non-fish": metrics['precision_nonfish'],
            "Recall_non-fish": metrics['recall_nonfish'],
            "AUPRC": metrics['auprc'],
            "AUROC": metrics['auroc']
        })

        with open(OUTPUT_DIR / "training_metrics_model_random_weights.json", "w") as f:
            json.dump(history, f, indent=4)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: F1-score did not improve for {patience} epochs")
            break

    print("\nTraining finished.")
    print(f"Best validation F1-score: {best_val_f1:.4f}")

    return f1_scores, precision_scores, recall_scores


# =========================
# Hyperparameter search
# =========================
def hyperparameter_search(device, data_dir):
    """
    Perform grid search over hyperparameters and save results.
    Input:
        device: torch device ('cpu' or 'cuda')
        data_dir: Path to dataset
    Output:
        None (results saved to 'hyperparameter_results.csv')
    """
    results = []

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    pos_weights = [1, 2, 5, 10]
    lrs = [5e-3, 1e-3, 5e-4, 1e-4]
    num_epochs = 20

    batch_size = 128 if device.type == "cuda" else 32
    experiment_id = 0

    for threshold in thresholds:
        for pos_weight in pos_weights:
            for lr in lrs:
                experiment_id += 1
                print("\n===============================")
                print(f"Experiment {experiment_id}")
                print(f"threshold={threshold}, pos_weight={pos_weight}, lr={lr}")

                # Load data
                train_loader_tmp, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
                train_dataset = train_loader_tmp.dataset
                train_loader = get_train_loader(train_dataset, batch_size=batch_size)

                # Create model
                model, criterion, optimizer = get_model(device=device, lr=lr, pos_weight=pos_weight)

                # Train
                f1_scores, precision_scores, recall_scores = run_training(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    num_epochs=num_epochs,
                    patience=10,
                    threshold=threshold,
                    model_name=f"best_model_exp_{experiment_id}.pth"
                )

                best_f1 = max(f1_scores)
                results.append({
                    "experiment": experiment_id,
                    "threshold": threshold,
                    "pos_weight": pos_weight,
                    "lr": lr,
                    "batch_size": batch_size,
                    "best_f1": best_f1
                })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values("best_f1", ascending=False)
    df.to_csv("hyperparameter_results.csv", index=False)

    print("\n===== BEST RESULTS =====")
    print(df.head(10))


# =========================
# Train final model with chosen hyperparameters
# =========================
def train_final_model(device, data_dir, threshold, pos_weight, lr, num_epochs, patience, batch_size):
    """
    Train the final model with chosen hyperparameters and save it.
    Input:
        device: torch device ('cpu' or 'cuda')
        data_dir: Path to dataset
        threshold: float, decision threshold for positive class 
        pos_weight: float, weight for positive class in BCE loss
        lr: float, learning rate for optimizer
        num_epochs: int, maximum number of epochs to train
        patience: int, number of epochs to wait for improvement before early stopping
        batch_size: int, batch size for training and validation DataLoaders       
    Output:
        None (model saved to 'model_random_weights.pt')
    """
    


    print("\n===== FINAL MODEL TRAINING =====")
    print(f"threshold = {threshold}")
    print(f"pos_weight = {pos_weight}")
    print(f"lr = {lr}")
    print(f"batch_size = {batch_size}")
    print(f"num_epochs = {num_epochs}")

    # Load data
    train_loader_tmp, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    train_dataset = train_loader_tmp.dataset
    train_loader = get_train_loader(train_dataset, batch_size=batch_size)

    # Create model
    model, criterion, optimizer = get_model(
        device=device,
        lr=lr,
        pos_weight=pos_weight
    )

    # Train
    f1_scores, precision_scores, recall_scores = run_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        patience=patience,
        threshold=threshold,
        model_name="model_random_weights.pt"
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Best F1: {max(f1_scores):.4f}")


# =========================
# Main function
# =========================
def main():
    # Detect device, set data path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    data_dir = Path("data/RODI-DATA/RODI-DATA/Train")
    
    # Set hyperparameters for final model training, train final model
    threshold = 0.75
    pos_weight = 1
    lr = 0.005
    num_epochs = 100
    patience = 10
    batch_size = 128 if DEVICE.type == "cuda" else 32
    
    train_final_model(DEVICE, data_dir, threshold, pos_weight, lr, num_epochs, patience,batch_size)


if __name__ == "__main__":
    main()