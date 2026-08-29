import torch
import torch.nn as nn
from torchvision import models


def get_model(config):
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if config["models"]["squeezenet"]["pretrained"] else None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Conv2d(512, config["project"]["num_classes"], kernel_size=1)
    model.num_classes = config["project"]["num_classes"]
    for param in model.classifier[1].parameters():
        param.requires_grad = True
    return model