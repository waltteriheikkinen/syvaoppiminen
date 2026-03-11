# src/data/data_loader.py

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
import numpy as np
import torch
from PIL import Image

class FineTunedDataset(torch.utils.data.Dataset):
    """
    Wrapper ImageFolder-datasetin ympärille, joka muuntaa kaikki luokat
    binary-luokiksi: 1 = fish, 0 = not fish
    """
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
        return img, label

def create_weighted_sampler(dataset):

    # haetaan alkuperäiset ImageFolder labelit
    targets = dataset.dataset.dataset.targets

    # subsetin indeksit
    indices = dataset.indices

    # muunnetaan binary labeliksi
    fish_idx = dataset.dataset.fish_class_idx

    labels = [1 if targets[i] == fish_idx else 0 for i in indices]

    class_sample_count = np.bincount(labels)
    class_weights = 1. / class_sample_count

    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, image_size=50, num_workers=4):
    """
    Luo train ja validation dataloaderit kuvien binary-luokitteluun (fish vs not fish)
    Kuvasuhde säilytetään Resize + Crop -kombinaatiolla.
    """

    # ======================
    # Transformit
    # ======================

    # Treeni: resize, satunnainen crop, augmentointi
    train_transforms = transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,       # Brightness ±20%
                           contrast=0.2,             # Contrast ±20%
                           saturation=0.2,           # Saturation ±20%
                           hue=0.05),                # Hue ±5%
        transforms.RandomAffine(degrees=15,          
                            translate=(0.1, 0.1)),   # Siirto ±10% x- ja y-suunnassa
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Validointi
    val_transforms = transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ======================
    # Dataset
    # ======================
    train_dataset_full = datasets.ImageFolder(
    root=data_dir,
    transform=train_transforms
    )

    val_dataset_full = datasets.ImageFolder(
    root=data_dir,
    transform=val_transforms
    )

    binary_train = FineTunedDataset(train_dataset_full)
    binary_val = FineTunedDataset(val_dataset_full)

    # ======================
    # Train/Val split
    # ======================
    dataset_size = len(binary_train)
    indices = torch.randperm(dataset_size)

    val_size = int(dataset_size * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(binary_train, train_indices)
    val_dataset = Subset(binary_val, val_indices)

    # Oma sämpleri
    sampler = create_weighted_sampler(train_dataset)

    # ======================
    # DataLoaders
    # ======================
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              sampler=sampler,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


class ResizeWithPadding:
    def __init__(self, target_size=50):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        
        # Skaalauskerroin
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize säilyttäen aspect ratio
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Luodaan musta taustakuva 50x50
        new_img = Image.new("RGB", (self.target_size, self.target_size))
        
        # Keskitetään kuva
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img
