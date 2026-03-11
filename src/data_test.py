import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun
from src.data_model import ResizeWithPadding

class TestDataset(Dataset):
    """
    Dataset mallin testausta varten.
    Lukee kaikki kuvat hakemistosta,
    käyttää samaa ResizeWithPadding -transformia kuin trainingissa.
    Palauttaa: (image_tensor, image_path)
    """

    def __init__(self, folder: str, image_size: int = 224):
        self.paths = list(Path(folder).rglob("*"))
        # Suodatetaan vain yleisimmät kuvat
        self.paths = [p for p in self.paths if p.suffix.lower() in {".jpg", ".jpeg"}]
        
        self.transform = transforms.Compose([
            ResizeWithPadding(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, str(path)