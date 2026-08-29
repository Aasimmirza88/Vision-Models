@echo off
echo ============================================
echo Flower Classification Training Pipeline
echo Using torch126 conda environment
echo ============================================
echo.

echo Activating torch126 environment...
call conda activate torch126

echo.
echo Checking dependencies...
python -c "import torch; import torchvision; import timm; import sklearn; import seaborn; import yaml; print('All dependencies OK')"

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some dependencies missing. Installing...
    pip install timm wandb scikit-learn pyyaml seaborn
)

echo.
echo Starting training...
echo Usage:
echo   run_training.bat all          - Train all models sequentially
echo   run_training.bat alexnet      - Train a specific model
echo.

if "%1"=="" (
    python monitor.py all
) else (
    python monitor.py %1
)

echo.
echo Training pipeline finished.