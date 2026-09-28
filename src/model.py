import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(num_classes: int) -> nn.Module:
    """
    Loads a pretrained ResNet18, freezes every layer except a newly added
    final layer sized for our classes. Only that new layer trains, which
    is what makes this fast and workable without a huge dataset.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model