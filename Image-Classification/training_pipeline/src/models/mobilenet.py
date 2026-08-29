import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if config["models"]["mobilenet_v2"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config["project"]["num_classes"])
    for param in model.classifier[1].parameters():
        param.requires_grad = True
    return model