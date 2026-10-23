@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
if not exist "%ROOT%\outputs" mkdir "%ROOT%\outputs"

set COUNT=0
set "PROBE_FILE=%TEMP%\subtitle_suite_probe_%RANDOM%.txt"
for /d %%D in ("%ROOT%\venvs\venv_*") do (
    set "ENV_NAME=%%~nxD"
    set "MODEL_ID=!ENV_NAME:venv_=!"
    if exist "%%~fD\Scripts\python.exe" (
        "%%~fD\Scripts\python.exe" "%ROOT%\app\main.py" --probe !MODEL_ID! >"%PROBE_FILE%" 2>nul
        if !errorlevel! EQU 0 (
            set /p MODEL_LABEL=<"%PROBE_FILE%"
            set /a COUNT+=1
            set "MODEL_ID_!COUNT!=!MODEL_ID!"
            set "MODEL_PY_!COUNT!=%%~fD\Scripts\python.exe"
            set "MODEL_LABEL_!COUNT!=!MODEL_LABEL!"
        )
    )
)
del "%PROBE_FILE%" >nul 2>&1

if %COUNT% EQU 0 (
    echo No usable model environment found.
    echo Run install.bat and install at least one model.
    pause
    exit /b 1
)

echo.
echo Available models:
echo.
set "KEYS="
for /l %%I in (1,1,%COUNT%) do (
    echo   [%%I] !MODEL_LABEL_%%I!
    set "KEYS=!KEYS!%%I"
)
echo.
choice /c !KEYS! /n /m "Select a model: "
set SEL=%errorlevel%

set "SEL_ID=!MODEL_ID_%SEL%!"
set "SEL_PY=!MODEL_PY_%SEL%!"

"!SEL_PY!" "%ROOT%\app\main.py" --model !SEL_ID!

if %errorlevel% neq 0 (
    echo.
    echo An error has occurred! Press any key to close...
    pause >nul
)
