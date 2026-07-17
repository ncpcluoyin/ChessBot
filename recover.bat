@echo off
cd /d "%~dp0"

set EPOCHS=200
if not "%TARGET_EPOCH%"=="" set EPOCHS=%TARGET_EPOCH%

echo =================================
echo   Recovery Training
echo   冻结骨干, 只训新头
echo =================================
echo.

.venv311\Scripts\python.exe -u -m src.main distill --data data\hf_supervised_samples --epochs %EPOCHS% --model data\models\model_sf.pt --resume --recover

echo.
pause
