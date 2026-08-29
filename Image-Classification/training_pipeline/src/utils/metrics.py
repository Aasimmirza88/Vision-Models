import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os


def compute_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    loss = torch.nn.functional.cross_entropy(outputs, labels).item()
    return accuracy, loss, preds.cpu().numpy(), labels.cpu().numpy()


def compute_class_metrics(outputs, labels, class_names):
    _, preds = torch.max(outputs, 1)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    report = classification_report(labels_np, preds_np, target_names=class_names, output_dict=True)
    return report


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_sample_predictions(model, val_loader, class_names, device, save_path, n=8):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(18, 8))

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                if images_so_far >= n:
                    break
                images_so_far += 1
                ax = plt.subplot(2, n // 2, images_so_far)
                img = inputs[j].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                color = "green" if preds[j] == labels[j] else "red"
                ax.set_title(f"Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}",
                             color=color, fontsize=9)
                ax.axis("off")
            if images_so_far >= n:
                break

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_results(model_name, history, report, confusion_mat_path, predictions_path, output_dir):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    import json
    results = {
        "model": model_name,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
        "best_epoch": history["val_acc"].index(max(history["val_acc"])) + 1 if history["val_acc"] else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "class_report": report,
    }
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return model_dir