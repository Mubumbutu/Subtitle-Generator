@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
chcp 65001 >nul
title SubtitleSuite - MOSS-Transcribe-Diarize Installer

echo.
echo ============================================================
echo SubtitleSuite - MOSS-Transcribe-Diarize Installation
echo ============================================================
echo.

echo [1/7] Checking Python...
set "PY_CMD=py -3"
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.12"
%PY_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Download Python 3.12 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] MOSS-Transcribe-Diarize requires Python 3.10 or newer, Python 3.12 is recommended.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('%PY_CMD% --version 2^>^&1') do set PY_VER=%%v
echo [OK] Using Python !PY_VER!

echo.
echo [2/7] Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv 2^>nul ^| findstr /R "^[0-9]"') do (
        set DRIVER_MAJOR=%%a
        goto DRIVER_DONE
    )
)
:DRIVER_DONE

if !GPU_FOUND! EQU 1 (
    echo [OK] NVIDIA GPU detected.
    echo Driver major version: !DRIVER_MAJOR!
) else (
    echo [INFO] No NVIDIA GPU detected.
)
echo.
echo Choose installation type:
echo.
echo [1] CPU only
echo [2] GPU ^(NVIDIA CUDA^)
echo.

:CHOICE
set USER_CHOICE=
set /p USER_CHOICE=Enter choice [1/2]: 
if "%USER_CHOICE%"=="1" goto CPU_MODE
if "%USER_CHOICE%"=="2" goto GPU_MODE
echo Invalid choice. Try again.
goto CHOICE

:CPU_MODE
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
goto CREATE_VENV

:GPU_MODE
if !GPU_FOUND! EQU 0 (
    echo [WARNING] No GPU detected - falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 570 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
    set TORCH_VARIANT=GPU CUDA 12.8
) else if !DRIVER_MAJOR! GEQ 560 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
    set TORCH_VARIANT=GPU CUDA 12.6
) else (
    echo [WARNING] Driver too old for the CUDA builds of PyTorch 2.8 or newer - falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
)
goto CREATE_VENV

:CREATE_VENV
echo.
echo [3/7] Selected mode: !TORCH_VARIANT!
echo.
set "VENV_DIR=%ROOT%\venvs\venv_moss"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
if exist "%VENV_DIR%" if not exist "%VENV_PY%" (
    echo [WARNING] Broken environment found - removing it.
    rmdir /s /q "%VENV_DIR%"
)
if exist "%VENV_PY%" (
    echo [INFO] Existing MOSS environment found.
    choice /c UR /n /m "[U]pdate it or [R]ecreate it from scratch? "
    if errorlevel 2 (
        rmdir /s /q "%VENV_DIR%"
        echo [OK] Old venv removed
    )
)
if not exist "%VENV_PY%" (
    echo [3/7] Creating virtual environment...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] venv created.
)
echo.

echo [4/7] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

echo.
echo [5/7] Installing PyTorch ^(!TORCH_VARIANT!^)...
"%VENV_PY%" -m pip install "torch>=2.8" "torchaudio>=2.8" --index-url !TORCH_INDEX_URL!
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

echo.
echo [6/7] Installing requirements and MOSS-Transcribe-Diarize...
"%VENV_PY%" -m pip install -r "%ROOT%\install\moss_requirements.txt"
if errorlevel 1 (
    echo [ERROR] Requirements installation failed.
    pause
    exit /b 1
)
echo [OK] Requirements installed.

echo.
echo [7/7] Verifying installation...
"%VENV_PY%" -c "import PyQt6, torch, transformers, moss_transcribe_diarize, librosa, sounddevice, soundfile, pysrt; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available()); print('transformers', transformers.__version__)"
if errorlevel 1 (
    echo [ERROR] Verification failed: required packages could not be imported.
    pause
    exit /b 1
)
echo [OK] Packages import correctly.
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found in system PATH - video input will not work^^!
) else (
    echo [OK] FFmpeg detected.
)
echo.
echo ============================================================
echo MOSS INSTALLATION COMPLETE ^(!TORCH_VARIANT!^)
echo The model weights are downloaded on first use.
echo ============================================================
echo.
pause
exit /b 0
