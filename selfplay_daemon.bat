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

echo ============================================================
echo   Self-Play Training
echo   Model:   %MODEL%
echo   Games:   %GAMES% per cycle
echo   Sims:    %SIMS%
echo   Train:   %EPOCHS% epochs per cycle
echo   Cycles:  %CYCLES%
echo ============================================================

for /l %%c in (1,1,%CYCLES%) do (
    echo.
    echo ===== Cycle %%c/%CYCLES% =====
    .venv311\Scripts\python.exe -u -m src.self_play ^
        --model "%MODEL%" ^
        --games %GAMES% ^
        --sims %SIMS% ^
        --train-epochs %EPOCHS% ^
        --workers %WORKERS%
    if errorlevel 1 (
        echo [Error in cycle %%c]
        pause
        exit /b 1
    )
)

echo.
echo [Self-play training complete]
pause
