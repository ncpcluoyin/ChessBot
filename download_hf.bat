@echo off
cd /d "%~dp0"

set POSITIONS=80000000
set OUTPUT=data\hf_supervised_samples

echo ============================================================
echo   Download HuggingFace Chess Evaluations
echo ============================================================
echo   Target:  %POSITIONS% positions total
echo   Output:  %OUTPUT%
echo ============================================================
echo.

.venv311\Scripts\python.exe scripts\download_hf_dataset.py ^
    --num-positions %POSITIONS% ^
    --output "%OUTPUT%" ^
    --resume

echo.
pause
