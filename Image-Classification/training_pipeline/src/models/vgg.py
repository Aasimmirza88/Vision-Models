import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if config["models"]["vgg16"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, config["project"]["num_classes"])
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model