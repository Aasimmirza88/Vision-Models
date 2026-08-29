import os
import sys
import json
import time
import gc
import traceback
import subprocess
import yaml
import torch
from datetime import datetime
from src.base_trainer import ModelTrainer


class TrainingMonitor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = self.config["project"]["output_dir"]
        self.logs_dir = self.config["project"]["logs_dir"]
        self.monitor_log = []
        self.env_ok = True

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _log(self, message, level="INFO"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self.monitor_log.append(entry)
        prefix = {"INFO": "[INFO]", "WARNING": "[WARN]", "ERROR": "[ERR]", "SUCCESS": "[OK]", "RETRY": "[RET]"}
        print(f"{prefix.get(level, '[LOG]')} {message}")

    def check_environment(self):
        self._log("Checking environment dependencies...")
        issues = []

        checks = {
            "torch": "torch",
            "torchvision": "torchvision",
            "timm": "timm",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "seaborn": "seaborn",
            "matplotlib": "matplotlib",
            "numpy": "numpy",
        }

        for module_name, pip_name in checks.items():
            try:
                __import__(module_name)
                self._log(f"  {module_name}: OK", "SUCCESS")
            except ImportError:
                issues.append(f"Missing: {module_name} (pip: {pip_name})")
                self._log(f"  {module_name}: MISSING", "ERROR")

        if not torch.cuda.is_available():
            issues.append("CUDA is not available - training will be on CPU")
            self._log("CUDA not available, will use CPU", "WARNING")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self._log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)", "SUCCESS")

        dataset_path = self.config["project"]["dataset_path"]
        if not os.path.exists(dataset_path):
            issues.append(f"Dataset not found at {dataset_path}")
            self._log(f"Dataset missing: {dataset_path}", "ERROR")
        else:
            self._log(f"Dataset found: {dataset_path}", "SUCCESS")

        if issues:
            self.env_ok = False
            self._log(f"Environment has {len(issues)} issue(s)", "WARNING")
        else:
            self._log("All environment checks passed", "SUCCESS")

        return self.env_ok

    def install_missing_deps(self):
        missing = []
        checks = {
            "torch": "torch",
            "torchvision": "torchvision",
            "timm": "timm",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "seaborn": "seaborn",
            "matplotlib": "matplotlib",
            "numpy": "numpy",
        }
        for module_name, pip_name in checks.items():
            try:
                __import__(module_name)
            except ImportError:
                missing.append(pip_name)

        if missing:
            self._log(f"Installing missing packages: {', '.join(missing)}", "WARNING")
            for pkg in missing:
                self._log(f"  Installing {pkg}...")
                result = subprocess.run(
                    ["conda", "run", "-n", "torch126", "pip", "install", pkg],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    self._log(f"  {pkg}: installed", "SUCCESS")
                else:
                    self._log(f"  {pkg}: FAILED - {result.stderr[:200]}", "ERROR")

    def rectify_oom_error(self, model_name, error_msg):
        self._log(f"Rectifying OOM for {model_name}...", "RETRY")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self._log("  Cleared CUDA cache", "INFO")

        current_bs = self.config["data"]["batch_size"]
        new_bs = max(1, int(current_bs * 0.5))
        self.config["data"]["batch_size"] = new_bs
        self._log(f"  Reduced batch_size: {current_bs} -> {new_bs}", "INFO")

        model_cfg = self.config["models"].get(model_name, {})
        if "batch_size" in model_cfg:
            new_model_bs = max(1, int(model_cfg["batch_size"] * 0.5))
            model_cfg["batch_size"] = new_model_bs
            self.config["data"]["batch_size"] = new_model_bs
            self._log(f"  Model-specific batch_size: {new_model_bs}", "INFO")

        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        return new_bs

    def rectify_runtime_error(self, model_name, error_msg):
        self._log(f"Rectifying runtime error for {model_name}...", "RETRY")

        if "out of memory" in error_msg.lower():
            return self.rectify_oom_error(model_name, error_msg)

        if "CUDA" in error_msg or "cuda" in error_msg:
            self._log("  CUDA error detected, switching to CPU", "WARNING")
            self.config["data"]["batch_size"] = 4
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)

        return self.config["data"]["batch_size"]

    def run_model_with_monitoring(self, model_name):
        self._log(f"{'='*60}")
        self._log(f"Starting monitored training for: {model_name}")
        self._log(f"{'='*60}")

        from src.base_trainer import ModelTrainer
        from src.models.alexnet import get_model as alexnet_fn
        from src.models.vgg import get_model as vgg_fn
        from src.models.resnet import get_model as resnet_fn
        from src.models.googlenet import get_model as googlenet_fn
        from src.models.mobilenet import get_model as mobilenet_fn
        from src.models.densenet import get_model as densenet_fn
        from src.models.squeezenet import get_model as squeezenet_fn
        from src.models.efficientnet import get_model as efficientnet_fn
        from src.models.nasnet import get_model as nasnet_fn
        from src.models.xception import get_model as xception_fn

        model_fns = {
            "alexnet": alexnet_fn,
            "vgg16": vgg_fn,
            "resnet50": resnet_fn,
            "googlenet": googlenet_fn,
            "mobilenet_v2": mobilenet_fn,
            "densenet121": densenet_fn,
            "squeezenet": squeezenet_fn,
            "efficientnet_b0": efficientnet_fn,
            "nasnet_large": nasnet_fn,
            "xception": xception_fn,
        }

        max_retries = self.config["failure_handling"]["max_retries"]
        retry_on_oom = self.config["failure_handling"]["retry_on_oom"]
        skip_on_failure = self.config["failure_handling"]["skip_on_failure"]

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self._log(f"Retry attempt {attempt}/{max_retries} for {model_name}", "RETRY")

            try:
                model_fn = model_fns.get(model_name)
                if model_fn is None:
                    raise ValueError(f"Unknown model: {model_name}")

                trainer = ModelTrainer(model_name, model_fn, self.config_path)
                success = trainer.train()

                status = trainer.get_status()
                if success:
                    self._log(f"{model_name} training COMPLETED successfully", "SUCCESS")
                    self._log(f"  Best Val Acc: {status['best_val_acc']:.4f}", "SUCCESS")
                    return True, status
                else:
                    self._log(f"{model_name} training FAILED: {status['failure_reason']}", "ERROR")

                    if status["failure_reason"] and "out of memory" in status["failure_reason"].lower():
                        if retry_on_oom and attempt < max_retries:
                            self.rectify_oom_error(model_name, status["failure_reason"])
                            continue

                    if skip_on_failure:
                        return False, status
                    raise RuntimeError(status["failure_reason"])

            except RuntimeError as e:
                error_msg = str(e)
                self._log(f"RuntimeError for {model_name} (attempt {attempt + 1}): {error_msg[:200]}", "ERROR")

                if "out of memory" in error_msg.lower() and retry_on_oom and attempt < max_retries:
                    self.rectify_oom_error(model_name, error_msg)
                    continue

                if attempt == max_retries:
                    self._log(f"All {max_retries + 1} attempts failed for {model_name}", "ERROR")
                    return False, {"failed": True, "failure_reason": error_msg}

                self.rectify_runtime_error(model_name, error_msg)

            except Exception as e:
                error_msg = str(e)
                self._log(f"Unexpected error for {model_name} (attempt {attempt + 1}): {error_msg[:200]}", "ERROR")

                # Don't retry on post-training errors (import errors, attribute errors, etc.)
                post_training_errors = ["cannot import name", "ImportError", "AttributeError", "ModuleNotFoundError"]
                is_post_training_error = any(err in error_msg for err in post_training_errors)

                if is_post_training_error:
                    self._log(f"Post-training error detected, not retrying: {error_msg[:100]}", "WARNING")
                    return False, {"failed": True, "failure_reason": error_msg, "post_training_error": True}

                if attempt == max_retries:
                    self._log(f"All {max_retries + 1} attempts failed for {model_name}", "ERROR")
                    return False, {"failed": True, "failure_reason": error_msg}

                self.rectify_runtime_error(model_name, error_msg)

        return False, {"failed": True, "failure_reason": "Unknown error"}

    def pre_download_all_weights(self):
        self._log("Pre-downloading pretrained weights for all models...")
        from src.models.alexnet import get_model as alexnet_fn
        from src.models.vgg import get_model as vgg_fn
        from src.models.resnet import get_model as resnet_fn
        from src.models.googlenet import get_model as googlenet_fn
        from src.models.mobilenet import get_model as mobilenet_fn
        from src.models.densenet import get_model as densenet_fn
        from src.models.squeezenet import get_model as squeezenet_fn
        from src.models.efficientnet import get_model as efficientnet_fn
        from src.models.nasnet import get_model as nasnet_fn
        from src.models.xception import get_model as xception_fn

        model_fns = {
            "alexnet": alexnet_fn,
            "vgg16": vgg_fn,
            "resnet50": resnet_fn,
            "googlenet": googlenet_fn,
            "mobilenet_v2": mobilenet_fn,
            "densenet121": densenet_fn,
            "squeezenet": squeezenet_fn,
            "efficientnet_b0": efficientnet_fn,
            "nasnet_large": nasnet_fn,
            "xception": xception_fn,
        }

        for model_name, model_fn in model_fns.items():
            self._log(f"  Downloading weights for {model_name}...")
            try:
                trainer = ModelTrainer(model_name, model_fn, self.config_path)
                trainer.download_and_store_weights()
                self._log(f"    {model_name}: OK", "SUCCESS")
            except Exception as e:
                self._log(f"    {model_name}: FAILED - {str(e)[:100]}", "ERROR")

        self._log("Weight pre-download complete", "SUCCESS")

    def run_all_models(self):
        self._log("Starting monitored training pipeline for all models")
        self._log(f"Environment: torch126 conda env")
        self._log(f"Hardware: GTX 1650 (4GB VRAM)")

        start_time = time.time()

        if not self.check_environment():
            self._log("Environment issues detected. Attempting to fix...", "WARNING")
            self.install_missing_deps()
            self.check_environment()

        self.pre_download_all_weights()

        results = {}
        succeeded = []
        failed = []

        model_names = [
            "alexnet", "vgg16", "resnet50", "googlenet", "mobilenet_v2",
            "densenet121", "squeezenet", "efficientnet_b0", "nasnet_large", "xception",
        ]

        for model_name in model_names:
            success, status = self.run_model_with_monitoring(model_name)
            results[model_name] = status

            if success:
                succeeded.append(model_name)
            else:
                failed.append(model_name)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        self._log(f"{'='*60}")
        self._log(f"PIPELINE COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self._log(f"Succeeded: {len(succeeded)}/{len(model_names)}")
        self._log(f"Failed: {len(failed)}/{len(model_names)}")

        if succeeded:
            self._log(f"Successful models: {', '.join(succeeded)}", "SUCCESS")
        if failed:
            self._log(f"Failed models: {', '.join(failed)}", "ERROR")

        summary_path = os.path.join(self.results_dir, "pipeline_summary.json")
        os.makedirs(self.results_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": elapsed,
                "succeeded": succeeded,
                "failed": failed,
                "results": results,
                "monitor_log": self.monitor_log,
            }, f, indent=2, default=str)
        self._log(f"Pipeline summary saved to {summary_path}", "SUCCESS")

        return results


if __name__ == "__main__":
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    monitor = TrainingMonitor(config_path)

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        monitor.run_all_models()
    elif len(sys.argv) > 1 and sys.argv[1] in [
        "alexnet", "vgg16", "resnet50", "googlenet", "mobilenet_v2",
        "densenet121", "squeezenet", "efficientnet_b0", "nasnet_large", "xception",
    ]:
        monitor.run_model_with_monitoring(sys.argv[1])
    else:
        print("Usage:")
        print("  python monitor.py all          - Run all models with monitoring")
        print("  python monitor.py <model_name> - Run a specific model with monitoring")
        print(f"\nAvailable models: alexnet, vgg16, resnet50, googlenet, mobilenet_v2, densenet121, squeezenet, efficientnet_b0, nasnet_large, xception")