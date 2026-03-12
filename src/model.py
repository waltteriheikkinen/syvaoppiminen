# src/model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model():
    """
    Mallin arkkitehtuuri. Ladataan resnet18 ja muokataan output kerros.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model(model_path="model.pt", device=None):
    """
    Ladataan mallin painot. Päivitettynä layer 3 ja 4.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def main():
    model = load_model("model.pt")
    print(model)


if __name__ == "__main__":
    main()