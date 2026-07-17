@echo off
cd /d "%~dp0"

set EPOCHS=200
if not "%TARGET_EPOCH%"=="" set EPOCHS=%TARGET_EPOCH%
set MODEL=data\models\model_recovered.pt

REM 从原始模型复制, 确保有骨干权重
if not exist "%MODEL%" copy data\models\model_sf.pt "%MODEL%" >nul

echo =================================
echo   Recovery Training
echo   冻结骨干, 只训新头
echo   输出: %MODEL%
echo   max_games=1000
echo =================================
echo.

.venv311\Scripts\python.exe -u -m src.main distill ^
    --data data\hf_supervised_samples ^
    --epochs %EPOCHS% ^
    --model %MODEL% ^
    --resume ^
    --recover ^
    --max-games 1000

echo.
pause
