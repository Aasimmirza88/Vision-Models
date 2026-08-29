import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if config["models"]["alexnet"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, config["project"]["num_classes"])
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model