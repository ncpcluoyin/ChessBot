@echo off
cd /d "%~dp0"

set MODEL=data\models\model_sf.pt
set GAMES=%SP_GAMES%
if "%GAMES%"=="" set GAMES=200
set SIMS=%SP_SIMS%
if "%SIMS%"=="" set SIMS=2000
set WORKERS=%SP_WORKERS%
if "%WORKERS%"=="" set WORKERS=12
set OUTPUT=data\self_play_games

echo ============================================================
echo   Self-Play Game Generator
echo ============================================================
echo   Model:   %MODEL%
echo   Games:   %GAMES%
echo   Sims:    %SIMS% per move
echo   Workers: %WORKERS%
echo   Output:  %OUTPUT%
echo ============================================================
echo.

.venv311\Scripts\python.exe -u -m src.self_play ^
    --model "%MODEL%" ^
    --games %GAMES% ^
    --sims %SIMS% ^
    --workers %WORKERS% ^
    --output "%OUTPUT%"

echo.
echo [Self-play finished. Games saved to %OUTPUT%]
pause
