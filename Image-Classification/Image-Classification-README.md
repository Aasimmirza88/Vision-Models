# Image Classification Architectures

A comparative computer vision study of **Image-classification architectures** trained and evaluated on the same **5-class flower dataset**. This section of the `Vision-Models` repository is organized architecture-first: each model folder keeps its notebook, trained weights, checkpoints, metrics, and visual evaluation artifacts together, while `training_pipeline/` contains the shared reproducible training framework used across models.

## Dataset

The experiments use a five-class flower classification dataset with the following classes:

- Lilly
- Lotus
- Orchid
- Sunflower
- Tulip

The dataset itself is available on kaggle 
Link-https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset

## Architectures

| Architecture | Folder | Main idea |
|---|---|---|
| AlexNet | `Alex_NET/` | Early deep CNN using stacked convolutions, ReLU, pooling, dropout, and fully connected layers |
| VGG16 | `VGG/` | Deep uniform network built primarily from repeated `3x3` convolutions |
| GoogLeNet / Inception v1 | `GoogleNET/` | Parallel multi-scale convolution branches inside Inception modules |
| ResNet50 | `ResNET/` | Residual connections that make substantially deeper networks easier to optimize |
| SqueezeNet | `SqueezeNET/` | Parameter-efficient Fire modules using squeeze and expand operations |
| MobileNetV2 | `MobileNET/` | Depthwise separable convolutions with inverted residuals and linear bottlenecks |
| DenseNet121 | `DenseNET/` | Dense feature reuse through concatenated connections between layers |
| EfficientNet-B0 | `EfficientNET/` | MBConv blocks with compound scaling of depth, width, and resolution |
| NASNet-Large | `NasNET/` | Neural-architecture-search-derived normal and reduction cells |
| Xception | `Xception/` | Depthwise separable convolutions used as an extreme form of Inception |

## Repository Structure

```text
Image-Classification/
|
|-- README.md
|
|-- Alex_NET/
|   |-- pretrained_weights/
|   |-- AlexNet_PyTorch.ipynb
|   |-- best_model.pth
|   |-- checkpoint_epoch_15.pth
|   |-- checkpoint_epoch_30.pth
|   |-- confusion_matrix.png
|   |-- history.json
|   |-- profiling.json
|   |-- results.json
|   |-- sample_predictions.png
|   `-- training_curves.png
|
|-- DenseNET/
|-- EfficientNET/
|-- GoogleNET/
|-- MobileNET/
|-- NasNET/
|-- ResNET/
|-- SqueezeNET/
|-- VGG/
|-- Xception/
|
`-- training_pipeline/
    |-- README.md
    |-- config.yaml
    |-- monitor.py
    |-- train_all.py
    |-- train_alexnet.py
    |-- train_densenet.py
    |-- train_efficientnet.py
    |-- train_googlenet.py
    |-- train_mobilenet.py
    |-- train_nasnet.py
    |-- train_resnet.py
    |-- train_squeezenet.py
    |-- train_vgg.py
    |-- train_xception.py
    |-- run_clean.bat
    |-- run_training.bat
    |-- src/
    |   |-- base_trainer.py
    |   |-- models/
    |   `-- utils/
    `-- results/
```

The architecture directories follow the same general pattern as `Alex_NET/`: a model notebook plus the trained model, checkpoints, training history, profiling output, confusion matrix, training curves, and sample predictions.

## Architecture Folders

Each architecture directory is intended to be self-contained and combines three parts of the experiment:

**Architecture study**  
The Jupyter notebook documents and runs the architecture on the flower dataset.

**Trained artifacts**  
`best_model.pth` stores the best trained checkpoint for the experiment. Intermediate checkpoints are retained where available. Model weights are tracked with **Git LFS** because several checkpoints are too large for normal Git storage.

**Evaluation artifacts**  
The generated JSON and PNG files preserve the quantitative and visual outcome of the run:

```text
history.json
profiling.json
results.json
confusion_matrix.png
training_curves.png
sample_predictions.png
```

This makes it possible to inspect a model's implementation, training behavior, efficiency profile, final metrics, confusion matrix, and example predictions from the same folder.

## Shared Training Pipeline

`training_pipeline/` contains the common training and evaluation framework used for all ten architectures.

The pipeline provides:

- a common configuration through `config.yaml`
- sequential or individual model training
- a shared `ModelTrainer`
- dataset loading and transforms
- deterministic/reproducible settings
- training and validation loops
- confusion-matrix generation
- training-curve generation
- sample prediction visualization
- model profiling
- failure handling and retry logic
- per-model result export

The model implementations are centralized under:

```text
training_pipeline/src/models/
```

and reusable utilities are kept under:

```text
training_pipeline/src/utils/
```

The monitoring entrypoint also checks the environment before training and can retry failed runs with reduced batch size when an out-of-memory error occurs.

For the complete pipeline documentation, see [`training_pipeline/README.md`](training_pipeline/README.md).

## Environment Used

The recorded training pipeline was executed with:

```text
Conda environment : torch126
Python            : 3.10
CUDA              : 12.6
GPU               : NVIDIA GeForce GTX 1650 - 4 GB VRAM
PyTorch           : 2.8.0+cu126
Torchvision       : 0.23.0+cu126
```

## Running the Training Pipeline

Move into the shared pipeline directory:

```bash
cd Image-Classification/training_pipeline
```

Activate the environment:

```bash
conda activate torch126
```

### Train all architectures with monitoring

```bash
python monitor.py all
```

### Train one architecture

```bash
python monitor.py alexnet
python monitor.py resnet50
python monitor.py efficientnet_b0
```

The individual model scripts can also be executed directly:

```bash
python train_alexnet.py
python train_resnet.py
python train_xception.py
```

### Run all models without the monitoring wrapper

```bash
python train_all.py all
```

## Training Workflow

The per-model training workflow covers the complete experiment lifecycle:

1. experiment objective and configuration
2. environment and reproducibility setup
3. dataset loading and exploration
4. preprocessing and augmentation
5. architecture construction
6. loss and optimizer configuration
7. training
8. validation
9. model checkpointing
10. metric generation
11. confusion-matrix generation
12. sample predictions
13. model profiling
14. result serialization

The pipeline README documents the full implementation flow in more detail.

## Outputs

For each model, the experiment preserves both the trained network and the evidence used to evaluate it.

| Artifact | Purpose |
|---|---|
| `best_model.pth` | Best trained model checkpoint |
| `checkpoint_epoch_*.pth` | Intermediate training checkpoints |
| `history.json` | Epoch-wise training history |
| `results.json` | Final evaluation results |
| `profiling.json` | Model profiling statistics |
| `confusion_matrix.png` | Class-level prediction behavior |
| `training_curves.png` | Training and validation progression |
| `sample_predictions.png` | Qualitative model predictions |

## Scope of This Study

The goal of this section is not only to obtain classification accuracy, but to study how major CNN design ideas evolved and how those architectural choices behave under a common image-classification problem.

The collection spans several important stages in CNN development:

```text
AlexNet
   |
   v
VGG16
   |
   v
GoogLeNet
   |
   v
ResNet50
   |
   +-------------------+
   |                   |
   v                   v
SqueezeNet         DenseNet121
   |                   |
   v                   v
MobileNetV2       EfficientNet-B0
                       |
             +---------+---------+
             |                   |
             v                   v
        NASNet-Large          Xception
```

Together, the experiments provide a practical comparison of architectural depth, parameter efficiency, feature reuse, residual learning, multi-scale processing, depthwise separable convolution, compound scaling, and neural architecture search.

## Parent Repository

This module is part of **Vision-Models**, a growing collection of computer vision architectures, experiments, and benchmarks covering image classification and future work in detection, segmentation, vision transformers, multimodal vision, and efficient deployment.
