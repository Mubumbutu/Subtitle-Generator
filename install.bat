@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title SubtitleSuite - Installer
if not exist "venvs" mkdir "venvs"
if not exist "outputs" mkdir "outputs"

:MENU
cls
echo ============================================================
echo SubtitleSuite - Installer
echo ============================================================
echo.
echo Each model is installed in its own isolated environment.
echo.
echo [1] WhisperX
echo [2] MOSS-Transcribe-Diarize
echo [3] Parakeet-TDT-0.6B-v3
echo [Q] Quit
echo.
choice /c 123Q /n /m "Select an option: "
if errorlevel 4 goto END
if errorlevel 3 goto PARAKEET
if errorlevel 2 goto MOSS
call "%~dp0install\whisperx_install.bat"
goto MENU

:MOSS
call "%~dp0install\moss_install.bat"
goto MENU

:PARAKEET
call "%~dp0install\parakeet_install.bat"
goto MENU

:END
endlocal
exit /b 0
