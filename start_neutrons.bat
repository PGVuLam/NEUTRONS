@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay moi truong Python .venv.
    echo Hay tao moi truong va cai dat requirements.txt truoc.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m neutrons.desktop
if errorlevel 1 pause
