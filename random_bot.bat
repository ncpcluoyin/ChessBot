@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
.venv311\Scripts\python.exe test\random_bot.py
pause