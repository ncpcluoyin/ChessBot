@echo off
cd /d "%~dp0"

set NEW=data\models\model_sf.pt
set OLD=data\models\model_sf_old.pt
set GAMES=20
set SIMS=400

echo =============================
echo   New vs Old Model
echo   Games: %GAMES%  Sims: %SIMS%
echo =============================

.venv311\Scripts\python.exe -u tests\eval_old_new.py ^
    --new "%NEW%" --old "%OLD%" ^
    --games %GAMES% --sims %SIMS%

pause
