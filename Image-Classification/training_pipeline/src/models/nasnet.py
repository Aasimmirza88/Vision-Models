import torch
import torch.nn as nn


def get_model(config):
    try:
        import timm
        model = timm.create_model("nasnetalarge", pretrained=config["models"]["nasnet_large"]["pretrained"], num_classes=config["project"]["num_classes"])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        return model
    except Exception:
        from torchvision import models
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, config["project"]["num_classes"])
        return model