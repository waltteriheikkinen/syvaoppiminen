import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun

from src.fish_data import get_dataloaders

def imshow(img, title=None):
    """
    Näyttää yhden kuvan tensorista
    """
    # Denormalisoi
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()



def main():
    data_dir = r"C:\Users\miika\Desktop\koulujutut\deeplearning\syvaoppiminen\data\RODI-DATA\RODI-DATA\Train"
    train_loader, val_loader = get_dataloaders(data_dir, image_size=50, batch_size=32)

    # Tulostetaan shape ja muutama esimerkkikuva treenistä
    for images, labels in train_loader:
        print('train_loader:')
        print(images.shape, labels.shape)  # esim [32, 3, 50, 50]
        
        # Näytetään 4 ensimmäistä kuvaa
        for i in range(4):
            imshow(images[i], title=f'Label: {labels[i].item()}')
        break

    # Sama validaatiolle
    for images, labels in val_loader:
        print('val_loader:')
        print(images.shape, labels.shape)
        for i in range(4):
            imshow(images[i], title=f'Label: {labels[i].item()}')
        break


if __name__ == "__main__":
    main()