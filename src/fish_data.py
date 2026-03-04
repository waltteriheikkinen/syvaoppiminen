# src/data/data_loader.py

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

class FishBinaryDataset(torch.utils.data.Dataset):
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

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, image_size=224, num_workers=4):
    """
    Luo train ja validation dataloaderit kuvien binary-luokitteluun (fish vs not fish)
    """
    # ======================
    # Transformit
    # ======================
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ======================
    # Dataset
    # ======================
    full_dataset = datasets.ImageFolder(root=f"{data_dir}",
                                        transform=train_transforms)
    binary_dataset = FishBinaryDataset(full_dataset)

    # ======================
    # Train/Val split
    # ======================
    val_size = int(len(binary_dataset) * val_split)
    train_size = len(binary_dataset) - val_size
    train_dataset, val_dataset = random_split(binary_dataset, [train_size, val_size])

    # Muutetaan val_datasetin transformit ilman augmentaatiota
    val_dataset.dataset.dataset.transform = val_transforms

    # ======================
    # DataLoaders
    # ======================
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader