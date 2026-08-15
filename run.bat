@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Please run:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
python launcher.py
pause
