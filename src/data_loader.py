# src/data/data_loader.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=4):
    """
    Luo train ja validation dataloaderit kuvien luokitteluun.

    Args:
        data_dir (str): polku data-kansioon, jossa train/ ja val/ alikansiot
        batch_size (int): mini-batch koko
        image_size (int): kuvien resize koko
        num_workers (int): parallelism loaderille

    Returns:
        train_loader, val_loader
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
    # Datasets
    # ======================
    train_dataset = datasets.ImageFolder(root=f"{data_dir}",
                                        transform=train_transforms)
    #val_dataset = datasets.ImageFolder(root=f"{data_dir}/val",
    #                                   transform=val_transforms)

    # ======================
    # DataLoaders
    # ======================
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    
    #val_loader = DataLoader(val_dataset,
    #                        batch_size=batch_size,
    #                        shuffle=False,
    #                        num_workers=num_workers)

    return train_loader#, val_loader
