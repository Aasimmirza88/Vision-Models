import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if config["models"]["googlenet"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, config["project"]["num_classes"])
    for param in model.fc.parameters():
        param.requires_grad = True
    return model