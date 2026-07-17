@echo off
cd /d "%~dp0"

set EPOCHS=%TARGET_EPOCH%
if "%EPOCHS%"=="" set EPOCHS=200
set MODEL=data\models\model_sf.pt

echo =================================
echo   Recovery Training
echo   冻结骨干, 只训新头
echo =================================
echo.

.venv311\Scripts\python.exe -u -m src.main distill ^
    --data data\hf_supervised_samples ^
    --epochs %EPOCHS% ^
    --model %MODEL% ^
    --resume ^
    --recover

echo.
pause
