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
set DATA=data\self_play_games

echo ============================================================
echo   Self-Play Training
echo   Model:   %MODEL%
echo   Games:   %GAMES% per cycle, %SIMS% sims
echo   Train:   %EPOCHS% epochs per cycle
echo   Cycles:  %CYCLES%
echo ============================================================

for /l %%c in (1,1,%CYCLES%) do (
    echo.
    echo ===== Cycle %%c/%CYCLES% =====
    echo --- Generating games ---
    .venv311\Scripts\python.exe -u -m src.self_play ^
        --model "%MODEL%" ^
        --games %GAMES% ^
        --sims %SIMS% ^
        --workers %WORKERS% ^
        --output "%DATA%"
    if errorlevel 1 goto err

    echo --- Training ---
    .venv311\Scripts\python.exe -u src\self_play_train.py ^
        --model "%MODEL%" ^
        --data "%DATA%" ^
        --epochs %EPOCHS%
    if errorlevel 1 goto err
)

echo.
echo [Self-play training complete]
pause
exit /b 0

:err
echo [Error in cycle %%c]
pause
exit /b 1
