import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Lisää projektin juuri polkuun

from src.fish_data import get_dataloaders



def main():
    data_dir = r"C:\Users\waltteri\projects\kurssit\syvaoppiminen\project_work\data\RODI-DATA\RODI-DATA\Train"
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=32)

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break


if __name__ == "__main__":
    main()