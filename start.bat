@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%venv\Scripts\activate.bat"

"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%subtitle_generator.py"

if %errorlevel% neq 0 (
    echo.
    echo An error has occurred! Press any key to close...
    pause >nul
)