import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


def validate_with_metrics(model, val_loader, criterion, device, threshold=0.5):
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