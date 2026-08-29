import torch
import torch.nn as nn
import timm


def get_model(config):
    model = timm.create_model("xception", pretrained=config["models"]["xception"]["pretrained"], num_classes=config["project"]["num_classes"])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model