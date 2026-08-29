import os
import sys
import json
import yaml
import traceback
import subprocess


MODEL_REGISTRY = {
    "alexnet": ("src.models.alexnet", "get_model"),
    "vgg16": ("src.models.vgg", "get_model"),
    "resnet50": ("src.models.resnet", "get_model"),
    "googlenet": ("src.models.googlenet", "get_model"),
    "mobilenet_v2": ("src.models.mobilenet", "get_model"),
    "densenet121": ("src.models.densenet", "get_model"),
    "squeezenet": ("src.models.squeezenet", "get_model"),
    "efficientnet_b0": ("src.models.efficientnet", "get_model"),
    "nasnet_large": ("src.models.nasnet", "get_model"),
    "xception": ("src.models.xception", "get_model"),
}


def run_single_model(model_name, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    print(f"\n{'='*60}")
    print(f"Starting training for: {model_name}")
    print(f"{'='*60}")

    from src.base_trainer import ModelTrainer
    module_path, fn_name = MODEL_REGISTRY[model_name]
    module = __import__(module_path, fromlist=[fn_name])
    model_fn = getattr(module, fn_name)

    trainer = ModelTrainer(model_name, model_fn, config_path)
    success = trainer.train()

    status = trainer.get_status()
    print(f"\n[{model_name}] Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Best Val Acc: {status['best_val_acc']}")
    print(f"  Epochs Completed: {status['epochs_completed']}")
    if status["failed"]:
        print(f"  Failure Reason: {status['failure_reason']}")

    return success, status


def run_all_models(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    failure_config = config.get("failure_handling", {})
    max_retries = failure_config.get("max_retries", 2)
    skip_on_failure = failure_config.get("skip_on_failure", True)

    print("\nPre-downloading pretrained weights for all models...")
    for model_name in MODEL_REGISTRY:
        print(f"  Downloading weights for {model_name}...")
        try:
            module_path, fn_name = MODEL_REGISTRY[model_name]
            module = __import__(module_path, fromlist=[fn_name])
            model_fn = getattr(module, fn_name)
            from src.base_trainer import ModelTrainer
            trainer = ModelTrainer(model_name, model_fn, config_path)
            trainer.download_and_store_weights()
        except Exception as e:
            print(f"    {model_name}: FAILED - {e}")

    print("Weight pre-download complete.\n")

    results = {}
    failed_models = []
    succeeded_models = []

    for model_name in MODEL_REGISTRY:
        print(f"\n{'#'*60}")
        print(f"# Processing model: {model_name}")
        print(f"{'#'*60}")

        success = False
        retries = 0
        current_batch_size = None

        while retries <= max_retries:
            try:
                success, status = run_single_model(model_name, config_path)
                if success:
                    break
            except Exception as e:
                print(f"\n[ERROR] Unexpected error for {model_name} (attempt {retries + 1}): {e}")
                traceback.print_exc()

            retries += 1

            if retries <= max_retries:
                print(f"Retrying {model_name} (attempt {retries + 1}/{max_retries + 1})...")
                if os.path.exists("config.yaml"):
                    with open("config.yaml", "r") as f:
                        cfg = yaml.safe_load(f)
                    current_bs = cfg["data"]["batch_size"]
                    new_bs = max(1, int(current_bs * (failure_config.get("retry_batch_size_reduction", 0.5)) ** retries))
                    cfg["data"]["batch_size"] = new_bs
                    with open("config.yaml", "w") as f:
                        yaml.dump(cfg, f)
                    print(f"  Reduced batch_size to {new_bs}")

        if not success:
            failed_models.append(model_name)
            results[model_name] = {"status": "FAILED", "reason": "All retries exhausted"}
            if skip_on_failure:
                print(f"\n[SKIP] {model_name} failed after {max_retries + 1} attempts. Moving to next model...")
                continue
        else:
            succeeded_models.append(model_name)
            results[model_name] = {"status": "SUCCESS", "best_val_acc": status["best_val_acc"]}

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(MODEL_REGISTRY)}")
    print(f"Succeeded: {len(succeeded_models)}")
    print(f"Failed: {len(failed_models)}")
    print(f"\nSuccessful models:")
    for m in succeeded_models:
        print(f"  - {m}: {results[m].get('best_val_acc', 'N/A')}")
    print(f"\nFailed models:")
    for m in failed_models:
        print(f"  - {m}: {results[m].get('reason', 'Unknown')}")

    summary_path = os.path.join("results", "training_summary.json")
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return results


def check_environment():
    issues = []

    if not torch.cuda.is_available():
        issues.append("CUDA is not available - training will be on CPU (very slow)")
    else:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    oom_env = os.environ.get("TRAINING_OOM_LIMIT", None)
    if oom_env:
        print(f"WARNING: TRAINING_OOM_LIMIT env var is set: {oom_env}")

    skip_env = os.environ.get("SKIP_MODEL", None)
    if skip_env:
        print(f"WARNING: SKIP_MODEL env var is set: {skip_env}")

    if not os.path.exists("config.yaml"):
        issues.append("config.yaml not found in current directory")

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Flowers_dataset")
    if not os.path.exists(dataset_path):
        dataset_path = "E:/GIt_upload/Image_classification/Flowers_dataset"
    if not os.path.exists(dataset_path):
        issues.append(f"Dataset not found at {dataset_path}")

    return issues


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    print("Flower Classification Training Pipeline")
    print("=" * 60)

    env_issues = check_environment()
    if env_issues:
        print("\nEnvironment warnings:")
        for issue in env_issues:
            print(f"  - {issue}")

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all_models(config_path)
    elif len(sys.argv) > 1 and sys.argv[1] in MODEL_REGISTRY:
        run_single_model(sys.argv[1], config_path)
    else:
        print("Usage:")
        print("  python train_all.py all          - Train all models sequentially")
        print("  python train_all.py <model_name> - Train a specific model")
        print(f"\nAvailable models: {', '.join(MODEL_REGISTRY.keys())}")