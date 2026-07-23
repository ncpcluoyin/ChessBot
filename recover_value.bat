@echo off
cd "%~dp0"

set MODEL=data\models\model_sf.pt
set EPOCHS=10
set LR=0.0003
set WD=0.01
set LOSS_WEIGHT=3.0

echo ============================================================
echo   Value Head Recover - 3-class (cross-entropy)
echo   Model:  %MODEL%
echo   Epochs: %EPOCHS%  LR: %LR%  WD: %WD%
echo   Loss:   CE x %LOSS_WEIGHT%
echo ============================================================

.venv311\Scripts\python.exe -u src\recover_value_3class.py

pause
