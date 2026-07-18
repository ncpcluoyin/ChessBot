@echo off
cd /d "%~dp0"

set MODEL=data\models\model_sf.pt
set DATA=data\self_play_games
set EPOCHS=3
set LR=0.0001

echo ============================================================
echo   Self-Play Training
echo   Model:  %MODEL%
echo   Data:   %DATA%
echo   Epochs: %EPOCHS%
echo   LR:     %LR%
echo ============================================================

.venv311\Scripts\python.exe -u src\self_play_train.py ^
    --model "%MODEL%" ^
    --data "%DATA%" ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --max-samples 50000 ^
    --cleanup

echo.
echo [Done]
pause
