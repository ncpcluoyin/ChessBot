@echo off
cd /d "%~dp0"

set EPOCHS=200
if not "%TARGET_EPOCH%"=="" set EPOCHS=%TARGET_EPOCH%
set MODEL=data\models\model_sf.pt

echo =================================
echo   Value Head Recovery
echo   Dual LR: backbone=0.1x, value=1.0x
echo   max_games=1000
echo =================================
echo.

.venv311\Scripts\python.exe -u -m src.main distill ^
    --data data\hf_supervised_samples ^
    --epochs %EPOCHS% ^
    --model %MODEL% ^
    --resume ^
    --dual-lr ^
    --max-games 1000

echo.
pause
