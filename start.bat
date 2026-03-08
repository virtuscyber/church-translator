@echo off
title Church Live Translation
cd /d "%~dp0"
python run.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Make sure Python 3.11+ is installed.
    echo Download from https://www.python.org/downloads/
    echo.
    pause
)
