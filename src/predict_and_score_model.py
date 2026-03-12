import torch
import os
import csv
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import json
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_finetuned import get_model, DEVICE  # oletetaan, että mallin luonti löytyy tästä

# ======================
# Asetukset
# ======================
MODEL_PATH = Path("../outputs/model.pt")
IMAGE_DIR = Path("../data/RODI-DATA/RODI-DATA/Train")
OUTPUT_CSV = "../outputs/predictions_test.csv"
OUTPUT_METRICS = Path("../outputs/test_metrics_model.json")

BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224
THRESHOLD = 0.5

# ======================
# ResizeWithPadding
# ======================
class ResizeWithPadding:
    def __init__(self, target_size=50):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (self.target_size, self.target_size))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

# ======================
# Dataset wrapper
# ======================
class TestDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if 'fish' not in self.dataset.class_to_idx:
            raise ValueError("Dataset does not contain class 'fish'")
        self.fish_class_idx = self.dataset.class_to_idx['fish']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label = 1 if label == self.fish_class_idx else 0
        path = self.dataset.imgs[idx][0]  # full path
        return img, label, path

# ======================
# Pääohjelma
# ======================
def main():
    # Transformit
    test_transforms = transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset ja DataLoader
    dataset_full = datasets.ImageFolder(root=IMAGE_DIR, transform=test_transforms)
    dataset = TestDataSet(dataset_full)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Lataa malli
    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probs = []
    results = []

    print(f"Löytyi {len(dataset)} kuvaa. Aloitetaan testi...")

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for path, prob in zip(paths, probs):
                filename = os.path.basename(path)
                results.append((filename, prob.item()))
                print(f"{filename} -> {prob.item():.4f}")

    # ======================
    # METRICS
    # ======================
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs > THRESHOLD).astype(int)

    f1_fish = f1_score(all_labels, preds, pos_label=1)
    precision_fish = precision_score(all_labels, preds, pos_label=1)
    recall_fish = recall_score(all_labels, preds, pos_label=1)

    precision_nonfish = precision_score(all_labels, preds, pos_label=0)
    recall_nonfish = recall_score(all_labels, preds, pos_label=0)

    accuracy = accuracy_score(all_labels, preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    test_metrics = []

     # tallenna metrics historiaan
    test_metrics.append({
        "f1_fish":            f1_fish,
        "precision_fish":     precision_fish,
        "recall_fish":        recall_fish,
        "precision_non-fish": precision_nonfish,
        "Recall_non-fish":    recall_nonfish,
        "accuracy":           accuracy,
        "AUPRC":              auprc,
        "AUROC":              auroc
    })

    with open(OUTPUT_METRICS, "w") as f:
        json.dump(test_metrics, f, indent=4)


    print("\n--- Dataset metrics ---")
    print(f"F1 (fish): {f1_fish:.4f}")
    print(f"Precision (fish): {precision_fish:.4f}")
    print(f"Recall (fish): {recall_fish:.4f}")
    print(f"Precision (non-fish): {precision_nonfish:.4f}")
    print(f"Recall (non-fish): {recall_nonfish:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # ======================
    # CSV
    # ======================
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "probability"])
        writer.writerows(results)

    print(f"\nTesti valmis. Tulokset tallennettu tiedostoon {OUTPUT_CSV}")

        

if __name__ == "__main__":
    main()