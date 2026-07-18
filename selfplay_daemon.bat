@echo off
cd /d "%~dp0"

set MODEL=data\models\model_sf.pt
set GAMES=%SP_GAMES%
if "%GAMES%"=="" set GAMES=200
set SIMS=%SP_SIMS%
if "%SIMS%"=="" set SIMS=800
set WORKERS=%SP_WORKERS%
if "%WORKERS%"=="" set WORKERS=12
set EPOCHS=%SP_EPOCHS%
if "%EPOCHS%"=="" set EPOCHS=5
set CYCLES=%SP_CYCLES%
if "%CYCLES%"=="" set CYCLES=20
set LR=%SP_LR%
if "%LR%"=="" set LR=0.001

echo ============================================================
echo   Self-Play Training Pipeline
echo   Model:   %MODEL%
echo   Games:   %GAMES% per cycle, %SIMS% sims
echo   Train:   %EPOCHS% epochs per cycle, lr=%LR%
echo   Cycles:  %CYCLES%
echo ============================================================

for /l %%c in (1,1,%CYCLES%) do (
    echo.
    echo ===== Cycle %%c/%CYCLES% =====
    .venv311\Scripts\python.exe -u -m src.self_play ^
        --model "%MODEL%" ^
        --games %GAMES% ^
        --sims %SIMS% ^
        --workers %WORKERS% ^
        --output "data\self_play_games"
    if errorlevel 1 goto err

    .venv311\Scripts\python.exe -u src\self_play_train.py ^
        --model "%MODEL%" ^
        --data "data\self_play_games" ^
        --epochs %EPOCHS% ^
        --lr %LR%
    if errorlevel 1 goto err
)

echo.
echo [Self-play training complete]
pause
exit /b 0

:err
echo [Error]
pause
exit /b 1
