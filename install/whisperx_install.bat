@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
chcp 65001 >nul
title SubtitleSuite - WhisperX Installer

echo.
echo ============================================================
echo SubtitleSuite - WhisperX Installation
echo ============================================================
echo.

echo [1/8] Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('py -3 --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do set PY_MAJOR=%%a
echo [OK] Found: !PY_VER!
if not "!PY_MAJOR!"=="3" (
    echo [ERROR] Python 3.x required.
    pause
    exit /b 1
)

echo.
echo [2/8] Detecting NVIDIA GPU...
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
set INSTALL_CUDNN=0
goto CREATE_VENV

:GPU_MODE
if !GPU_FOUND! EQU 0 (
    echo [WARNING] No GPU detected - falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    set INSTALL_CUDNN=0
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    set INSTALL_CUDNN=1
) else if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    set INSTALL_CUDNN=1
) else if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    set INSTALL_CUDNN=1
) else (
    echo [WARNING] Driver too old - falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    set INSTALL_CUDNN=0
)
goto CREATE_VENV

:CREATE_VENV
echo.
echo [3/8] Selected mode: !TORCH_VARIANT!
echo.
set "VENV_DIR=%ROOT%\venvs\venv_whisperx"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
if exist "%VENV_DIR%" if not exist "%VENV_PY%" (
    echo [WARNING] Broken environment found - removing it.
    rmdir /s /q "%VENV_DIR%"
)
if exist "%VENV_PY%" (
    echo [INFO] Existing WhisperX environment found.
    choice /c UR /n /m "[U]pdate it or [R]ecreate it from scratch? "
    if errorlevel 2 (
        rmdir /s /q "%VENV_DIR%"
        echo [OK] Old venv removed
    )
)
if not exist "%VENV_PY%" (
    echo [3/8] Creating virtual environment...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] venv created.
)
echo.

echo [4/8] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

echo.
echo [5/8] Installing requirements...
"%VENV_PY%" -m pip install -r "%ROOT%\install\whisperx_requirements.txt" --extra-index-url !TORCH_INDEX_URL!
if errorlevel 1 (
    echo [ERROR] Requirements installation failed.
    pause
    exit /b 1
)
echo [OK] Requirements installed.

echo.
echo [6/8] Force-reinstalling PyTorch ^(!TORCH_VARIANT!^)...
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX_URL! --force-reinstall --no-deps
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK] PyTorch suite installed ^(torch, torchvision, torchaudio^).

echo.
if not "!INSTALL_CUDNN!"=="1" (
    echo [7/8] Skipping CUDA libs ^(CPU mode^).
    goto AFTER_CUDA
)
echo [7/8] Installing CUDA libraries...
"%VENV_PY%" -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
if errorlevel 1 (
    echo [ERROR] CUDA library installation failed.
    pause
    exit /b 1
)
echo [OK] CUDA libs installed.
:AFTER_CUDA

echo.
echo [8/8] Verifying installation...
"%VENV_PY%" -c "import PyQt6, torch, whisperx, librosa, sounddevice, soundfile, pysrt; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] Verification failed: required packages could not be imported.
    pause
    exit /b 1
)
echo [OK] Packages import correctly.
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found in system PATH^^!
) else (
    echo [OK] FFmpeg detected.
)
echo.
echo ============================================================
echo WHISPERX INSTALLATION COMPLETE ^(!TORCH_VARIANT!^)
echo ============================================================
echo.
pause
exit /b 0
