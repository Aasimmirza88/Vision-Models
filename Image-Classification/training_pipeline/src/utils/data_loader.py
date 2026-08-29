import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter


def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(config):
    input_size = config["data"]["input_size"]
    mean = config["data"]["normalize_mean"]
    std = config["data"]["normalize_std"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


def load_dataset(config):
    dataset_path = config["project"]["dataset_path"]
    train_transform, val_transform = get_transforms(config)

    full_dataset = datasets.ImageFolder(root=dataset_path)

    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx

    train_size = int(config["data"]["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config["reproducibility"]["seed"])
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset, class_names, class_to_idx


def get_data_loaders(config=None):
    if config is None:
        config = load_config()
    train_dataset, val_dataset, class_names, class_to_idx = load_dataset(config)

    batch_size = config["data"]["batch_size"]

    num_workers = config["data"]["num_workers"]
    if sys.platform == "win32":
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
    )

    return train_loader, val_loader, class_names, class_to_idx


def dataset_stats(dataset_path):
    class_names = sorted(os.listdir(dataset_path))
    stats = {}
    total = 0
    for cls in class_names:
        cls_path = os.path.join(dataset_path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
            stats[cls] = count
            total += count
    return stats, total