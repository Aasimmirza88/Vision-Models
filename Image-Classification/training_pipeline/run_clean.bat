@echo off
echo ============================================
echo Flower Classification Training Pipeline - CLEAN RESTART
echo ============================================
echo.

echo Clearing Python cache...
if exist "E:\GIt_upload\Image_classification\src\__pycache__" rmdir /s /q "E:\GIt_upload\Image_classification\src\__pycache__"
if exist "E:\GIt_upload\Image_classification\src\models\__pycache__" rmdir /s /q "E:\GIt_upload\Image_classification\src\models\__pycache__"
if exist "E:\GIt_upload\Image_classification\src\utils\__pycache__" rmdir /s /q "E:\GIt_upload\Image_classification\src\utils\__pycache__"
if exist "E:\GIt_upload\Image_classification\__pycache__" rmdir /s /q "E:\GIt_upload\Image_classification\__pycache__"

echo.
echo Activating torch126 environment...
call conda activate torch126

echo.
echo Starting training with monitor (will resume from checkpoints if exist)...
python monitor.py all

echo.
echo Training pipeline finished.
pause