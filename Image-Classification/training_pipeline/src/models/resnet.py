import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if config["models"]["resnet50"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, config["project"]["num_classes"])
    for param in model.fc.parameters():
        param.requires_grad = True
    return model