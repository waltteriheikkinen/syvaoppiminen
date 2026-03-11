import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

from src.model import get_model
from src.data_finetuned import get_dataloaders


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_finetuned_model.pt"
DATA_DIR = "../data/RODI-DATA/RODI-DATA/Train"

BATCH_SIZE = 32
IMAGE_SIZE = 224
VAL_SPLIT = 0.2

THRESHOLD = 0.5


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = get_model()
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def evaluate(model, dataloader):

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = (all_probs > THRESHOLD).astype(int)

    # --- fish metrics (positive class) ---
    f1_fish = f1_score(all_labels, preds, pos_label=1)
    precision_fish = precision_score(all_labels, preds, pos_label=1)
    recall_fish = recall_score(all_labels, preds, pos_label=1)

    # --- non-fish metrics ---
    precision_nonfish = precision_score(all_labels, preds, pos_label=0)
    recall_nonfish = recall_score(all_labels, preds, pos_label=0)

    # --- other metrics ---
    accuracy = accuracy_score(all_labels, preds)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    print("\nEvaluation results")
    print("-------------------")

    print(f"F1 (fish): {f1_fish:.4f}")

    print(f"Precision (fish): {precision_fish:.4f}")
    print(f"Recall (fish): {recall_fish:.4f}")

    print(f"Precision (non-fish): {precision_nonfish:.4f}")
    print(f"Recall (non-fish): {recall_nonfish:.4f}")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")


def main():

    print("Loading model...")
    model = load_model()

    print("Loading dataset...")
    _, val_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        image_size=IMAGE_SIZE
    )

    evaluate(model, val_loader)


if __name__ == "__main__":
    main()