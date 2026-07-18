@echo off
cd /d "%~dp0"

set MODEL=data\models\model_sf.pt
set SIMS=800
set WORKERS=24
set OUTPUT=data\self_play_games

echo ============================================================
echo   Self-Play Generator (continuous)
echo   Model:   %MODEL%
echo   Sims:    %SIMS%
echo   Workers: %WORKERS%
echo   Output:  %OUTPUT%
echo   Ctrl+C to stop
echo ============================================================

.venv311\Scripts\python.exe -u -m src.self_play ^
    --model "%MODEL%" ^
    --sims %SIMS% ^
    --workers %WORKERS% ^
    --output "%OUTPUT%"

pause
