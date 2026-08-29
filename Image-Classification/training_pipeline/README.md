# Flower Classification Training Pipeline

10-model ablation study on a 5-class flower dataset (Lilly, Lotus, Orchid, Sunflower, Tulip).

## Environment

- **Conda env**: `torch126` (Python 3.10, CUDA 12.6)
- **GPU**: GTX 1650 (4GB VRAM)
- **torch**: 2.8.0+cu126, **torchvision**: 0.23.0+cu126

## Project Structure

```
Image_classification/
├── config.yaml              # Hyperparameters and model configs
├── monitor.py               # Main orchestrator with error handling & rectification
├── train_all.py             # Alternative runner (all models sequentially)
├── train_<model>.py         # Individual per-model scripts (10 files)
├── src/
│   ├── base_trainer.py      # Base ModelTrainer class with failure handling
│   ├── models/
│   │   ├── alexnet.py       # AlexNet
│   │   ├── vgg.py           # VGG16
│   │   ├── resnet.py        # ResNet50
│   │   ├── googlenet.py     # GoogleNet
│   │   ├── mobilenet.py     # MobileNetV2
│   │   ├── densenet.py      # DenseNet121
│   │   ├── squeezenet.py    # SqueezeNet
│   │   ├── efficientnet.py  # EfficientNet-B0
│   │   ├── nasnet.py        # NASNet-Large (via timm)
│   │   └── xception.py      # Xception (via timm)
│   └── utils/
│       ├── data_loader.py   # Dataset loading & transforms
│       ├── metrics.py       # Confusion matrix, curves, predictions, results
│       ├── profiling.py     # Model profiling (params, FLOPs, memory, time)
│       └── reproducibility.py  # Seed & deterministic settings
├── results/                 # Per-model outputs (models, curves, confusion matrices)
└── logs/                    # Failure logs and monitoring output
```

## Usage

### Run all models with monitoring (recommended)
```bash
conda activate torch126
python monitor.py all
```

### Run a specific model
```bash
conda activate torch126
python monitor.py alexnet
# or
python train_alexnet.py
```

### Run all models without monitoring
```bash
conda activate torch126
python train_all.py all
```

## Per-Model Scripts

Each `train_<model>.py` script implements all 18 steps:
1. Project objective
2. Imports and environment
3. Reproducibility settings
4. Dataset description
5. Dataset loading
6. Data exploration
7. Transformations and augmentation
8. Model architecture
9. Layer-by-layer tensor shapes (optional)
10. Loss and optimizer
11. Training loop
12. Validation loop
13. Training results
14. Confusion matrix
15. Sample predictions
16. Failure cases
17. Model profiling
18. Conclusions

## Failure Handling

The `monitor.py` orchestrator:
- Checks environment dependencies before starting
- Monitors for OOM errors during training
- Automatically reduces batch size on OOM (50% reduction per retry)
- Retries up to `max_retries` (default: 2) per model
- Skips to next model if all retries fail
- Logs all failures to `logs/<model>/failure.json`
- Saves a pipeline summary to `results/pipeline_summary.json`

## Output

Each model's results are saved in `results/<model_name>/`:
- `best_model.pth` - Best model weights
- `confusion_matrix.png`
- `training_curves.png`
- `sample_predictions.png`
- `profiling.json`
- `results.json`
- `history.json`