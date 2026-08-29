import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


class ModelTrainer:
    def __init__(self, model_name, model_fn, config_path="config.yaml"):
        self.model_name = model_name
        self.model_fn = model_fn
        self.config = self._load_config(config_path)
        self._update_config_for_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(self.config["project"]["output_dir"], model_name)
        self.logs_dir = os.path.join(self.config["project"]["logs_dir"], model_name)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "epoch_times": [],
        }
        self.failed = False
        self.failure_reason = None
        self.current_batch_size = None

    def _load_config(self, config_path):
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _update_config_for_model(self):
        model_cfg = self.config["models"].get(self.model_name, {})
        self.config["_current_model"] = self.model_name
        if "batch_size" in model_cfg:
            self.config["data"]["batch_size"] = model_cfg["batch_size"]
        if "input_size" in model_cfg:
            self.config["data"]["input_size"] = model_cfg["input_size"]
        self.current_batch_size = self.config["data"]["batch_size"]

    def _get_model(self):
        return self.model_fn(self.config)

    def _get_data_loaders(self):
        from src.utils.data_loader import get_data_loaders
        return get_data_loaders(self.config)

    def _setup_training(self, model):
        criterion = nn.CrossEntropyLoss()
        lr = float(self.config["training"]["learning_rate"])
        wd = float(self.config["training"]["weight_decay"])
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config["training"]["scheduler_params"]["mode"],
            factor=float(self.config["training"]["scheduler_params"]["factor"]),
            patience=int(self.config["training"]["scheduler_params"]["patience"]),
        )
        return criterion, optimizer, scheduler

    def _train_one_epoch(self, model, criterion, optimizer, train_loader):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / len(train_loader), correct / total

    def _validate(self, model, criterion, val_loader):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.append(outputs.cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        return running_loss / len(val_loader), correct / total, all_preds, all_labels, all_outputs

    def download_and_store_weights(self):
        weights_dir = os.path.join(self.results_dir, "pretrained_weights")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            model = self.model_fn(self.config)
            if hasattr(model, "state_dict"):
                state_dict = model.state_dict()
                weights_path = os.path.join(weights_dir, "pretrained_weights.pth")
                torch.save(state_dict, weights_path)
                size_mb = os.path.getsize(weights_path) / (1024 * 1024)
                print(f"  Pretrained weights saved: {weights_path} ({size_mb:.1f} MB)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return weights_dir
        except Exception as e:
            print(f"  Warning: Could not save pretrained weights: {e}")
            return None

    def _find_last_checkpoint(self):
        checkpoints = []
        if os.path.isdir(self.results_dir):
            for f in os.listdir(self.results_dir):
                if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
                    epoch_num = int(f.replace("checkpoint_epoch_", "").replace(".pth", ""))
                    checkpoints.append((epoch_num, os.path.join(self.results_dir, f)))
        if not checkpoints:
            return None, None
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1]

    def _save_checkpoint(self, model, epoch):
        if epoch % 15 != 0 and epoch != self.config["training"]["epochs"]:
            return
        checkpoint_path = os.path.join(self.results_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def train(self):
        from src.utils.reproducibility import set_seed, set_deterministic
        set_seed(self.config["reproducibility"]["seed"])
        set_deterministic(self.config["reproducibility"]["deterministic"])

        print(f"\n  Downloading and storing pretrained weights for {self.model_name}...")
        self.download_and_store_weights()

        start_epoch = 0
        resume = self.config["training"].get("resume_from_checkpoint", False)
        if resume:
            last_epoch, checkpoint_path = self._find_last_checkpoint()
            if last_epoch is not None and last_epoch > 0:
                start_epoch = last_epoch
                print(f"  Resuming from checkpoint epoch {last_epoch}")

        try:
            model = self._get_model()
            model = model.to(self.device)
        except Exception as e:
            self.failed = True
            self.failure_reason = f"Model creation failed: {str(e)}"
            self._save_failure_log()
            return False

        try:
            train_loader, val_loader, class_names, class_to_idx = self._get_data_loaders()
        except Exception as e:
            self.failed = True
            self.failure_reason = f"Data loading failed: {str(e)}"
            self._save_failure_log()
            return False

        criterion, optimizer, scheduler = self._setup_training(model)

        if resume and start_epoch > 0:
            checkpoint_path = os.path.join(self.results_dir, f"checkpoint_epoch_{start_epoch}.pth")
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"  Loaded checkpoint from epoch {start_epoch}")

        best_val_acc = 0.0
        patience_counter = 0
        epochs = self.config["training"]["epochs"]

        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            try:
                train_loss, train_acc = self._train_one_epoch(model, criterion, optimizer, train_loader)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.failed = True
                    self.failure_reason = f"OOM at epoch {epoch + 1}: {str(e)}"
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    self._save_failure_log()
                    return False
                raise

            val_loss, val_acc, all_preds, all_labels, all_outputs = self._validate(
                model, criterion, val_loader
            )

            scheduler.step(val_acc)
            epoch_time = time.perf_counter() - epoch_start

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(epoch_time)

            print(f"[{self.model_name}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if self.config["logging"]["save_best_model"]:
                    torch.save(model.state_dict(), os.path.join(self.results_dir, "best_model.pth"))
            else:
                patience_counter += 1

            self._save_checkpoint(model, epoch + 1)

            if self.config["training"]["early_stopping"]["enabled"]:
                if patience_counter >= self.config["training"]["early_stopping"]["patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        try:
            self._save_final_results(model, all_preds, all_labels, all_outputs, class_names, class_to_idx)
        except Exception as e:
            print(f"  Warning: Error saving final results: {e}")

        return True

    def _save_final_results(self, model, all_preds, all_labels, all_outputs, class_names, class_to_idx):
        from src.utils.metrics import (
            compute_class_metrics, plot_confusion_matrix,
            plot_training_curves, plot_sample_predictions, save_results
        )
        from src.utils.profiling import profile_model

        report = compute_class_metrics(all_outputs, torch.tensor(all_labels), class_names)

        cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

        curves_path = os.path.join(self.results_dir, "training_curves.png")
        plot_training_curves(self.history, curves_path)

        _, val_loader, _, _ = self._get_data_loaders()
        preds_path = os.path.join(self.results_dir, "sample_predictions.png")
        plot_sample_predictions(model, val_loader, class_names, self.device, preds_path)

        if self.config["profiling"]["enabled"]:
            profile_model(model, self.config["data"]["input_size"], self.device, self.results_dir)

        save_results(self.model_name, self.history, report, cm_path, preds_path, self.config["project"]["output_dir"])

        with open(os.path.join(self.results_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n[{self.model_name}] Best Val Acc: {max(self.history['val_acc']):.4f} at epoch "
              f"{self.history['val_acc'].index(max(self.history['val_acc'])) + 1}")

    def _save_failure_log(self):
        failure_log = {
            "model": self.model_name,
            "failed": True,
            "reason": self.failure_reason,
            "timestamp": datetime.now().isoformat(),
            "batch_size": self.current_batch_size,
            "gpu_memory": None,
        }
        if torch.cuda.is_available():
            failure_log["gpu_memory"] = {
                "allocated_GB": torch.cuda.memory_allocated() / (1024 ** 3),
                "reserved_GB": torch.cuda.memory_reserved() / (1024 ** 3),
            }

        log_path = os.path.join(self.logs_dir, "failure.json")
        with open(log_path, "w") as f:
            json.dump(failure_log, f, indent=2, default=str)

        print(f"\n[{self.model_name}] FAILED: {self.failure_reason}")
        print(f"Failure log saved to {log_path}")

    def get_status(self):
        return {
            "model": self.model_name,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "best_val_acc": max(self.history["val_acc"]) if self.history["val_acc"] else None,
            "epochs_completed": len(self.history["train_loss"]),
        }