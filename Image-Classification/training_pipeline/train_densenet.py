import os
import sys
import json
import yaml
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.base_trainer import ModelTrainer
from src.models.densenet import get_model


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = "densenet121"
    print(f"Training {model_name} on flower dataset")
    print(f"Dataset: {config['project']['dataset_path']}")
    print(f"Classes: {config['project']['class_names']}")
    print(f"Device: {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")

    trainer = ModelTrainer(model_name, get_model, config_path)
    success = trainer.train()

    status = trainer.get_status()
    print(f"\n{'='*40}")
    print(f"DenseNet121 Training {'COMPLETED' if success else 'FAILED'}")
    print(f"{'='*40}")
    print(f"Best Val Acc: {status['best_val_acc']}")
    print(f"Epochs Completed: {status['epochs_completed']}")
    if status["failed"]:
        print(f"Failure Reason: {status['failure_reason']}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
