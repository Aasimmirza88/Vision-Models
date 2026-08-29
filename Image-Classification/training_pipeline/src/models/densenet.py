import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if config["models"]["densenet121"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, config["project"]["num_classes"])
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model