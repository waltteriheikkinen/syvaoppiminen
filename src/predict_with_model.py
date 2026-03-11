import torch
import sys
import os
import csv
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Lisää projektin juuri polkuun
from src.model import get_model, DEVICE
from src.data_test import TestDataset

MODEL_PATH = Path("../outputs/model.pt")
IMAGE_DIR = Path("../data/RODI-DATA/RODI-DATA/Train")   # hakemisto, jossa testikuvat
OUTPUT_CSV = "../outputs/predictions.csv"
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224


def load_trained_model():
    """Lataa koulutettu malli testiä varten"""
    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    # Luo dataset ja dataloader
    dataset = TestDataset(IMAGE_DIR, image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = load_trained_model()
    results = []

    print(f"Löytyi {len(dataset)} kuvaa. Aloitetaan testi...")

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)  # todennäköisyys luokalle 1

            for path, prob in zip(paths, probs):
                filename = os.path.basename(path)
                results.append((filename, prob.item()))
                print(f"{filename} -> {prob.item():.4f}")

    # Kirjoita CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "probability"])
        writer.writerows(results)

    print(f"Testi valmis. Tulokset tallennettu tiedostoon {OUTPUT_CSV}")


if __name__ == "__main__":
    main()