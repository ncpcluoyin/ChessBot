@echo off
REM ChessBot UCI - Windows
REM Usage: chessbot_fp.bat [model] [--intuition]
cd /d "%~dp0"
set MODEL=%~1
if "%MODEL%"=="" set MODEL=data\models\model_sf.pt
set INTUITION=%~2

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
.venv311\Scripts\python.exe -m src.main uci "%MODEL%" %INTUITION%