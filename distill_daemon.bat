@echo off
cd "%~dp0"

set TARGET=%TARGET_EPOCH%
if "%TARGET%"=="" set TARGET=1800
set MAX_GAMES=%MAX_GAMES%
if "%MAX_GAMES%"=="" set MAX_GAMES=2000
set WORKERS=%DISTILL_WORKERS%
if "%WORKERS%"=="" set WORKERS=0

set MODEL=data\models\model_sf.pt
set DATA_DIR=%DATA_DIR%
if "%DATA_DIR%"=="" set DATA_DIR=data\hf_supervised_samples

echo =================================
echo   Distill Training
echo =================================
echo   Data:   %DATA_DIR%
echo   Model:  %MODEL%
echo   Batch:  512  LR: 0.002
echo   Value:  3-class CE x 3.0, Policy: entropy x 1.2 + label smooth x 0.1, WD=1e-4
echo =================================
echo.

.venv311\Scripts\python.exe -u -m src.main distill ^
    --data "%DATA_DIR%" ^
    --epochs "%TARGET%" ^
    --model "%MODEL%" ^
    --max-games "%MAX_GAMES%" ^
    --workers "%WORKERS%" ^
    --castling-dir "data\castling_samples" ^
    --castling-ratio 0.2 ^
    --resume

echo.
echo [Training finished.]
pause
