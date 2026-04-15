@echo off
setlocal enabledelayedexpansion
:: ── 1. WORKING DIRECTORY LOCK (CRITICAL FIX) ──
cd /d "%~dp0"

chcp 65001 >nul
title Subtitle Generator - Bulletproof Installer (with RE-USE)
color 0A

echo.
echo ============================================================
echo Subtitle Generator - Bulletproof Installation
echo NVIDIA RE-USE : Multilingual Universal Speech Enhancement
echo ============================================================
echo.

:: ----------------------------------------------------------------
:: [1] Check Python
:: ----------------------------------------------------------------
echo [1/9] Checking Python...
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

:: ----------------------------------------------------------------
:: [2] GPU Detection
:: ----------------------------------------------------------------
echo.
echo [2/9] Detecting NVIDIA GPU...
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
echo [2] GPU (NVIDIA CUDA)
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

:: ----------------------------------------------------------------
:: [3] Create clean virtual environment
:: ----------------------------------------------------------------
:CREATE_VENV
echo.
echo [3/9] Selected mode: !TORCH_VARIANT!
echo.
echo [3/9] Creating clean virtual environment...
set "VENV_DIR=venv"
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] venv created.

:: Absolute venv Python path (CRITICAL FIX)
set "VENV_PY=%~dp0%VENV_DIR%\Scripts\python.exe"
echo.

:: ----------------------------------------------------------------
:: [4] Upgrade pip
:: ----------------------------------------------------------------
echo [4/9] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

:: ----------------------------------------------------------------
:: [5] Install requirements
:: ----------------------------------------------------------------
echo.
echo [5/9] Installing requirements...
"%VENV_PY%" -m pip install -r requirements.txt --extra-index-url !TORCH_INDEX_URL!
if errorlevel 1 (
    echo [ERROR] Requirements installation failed.
    pause
    exit /b 1
)
echo [OK] Requirements installed.

:: ----------------------------------------------------------------
:: [6] Force-reinstall PyTorch + torchaudio
:: ----------------------------------------------------------------
echo.
echo [6/9] Force-reinstalling PyTorch (!TORCH_VARIANT!)...
"%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX_URL! --force-reinstall --no-deps
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

:: ----------------------------------------------------------------
:: [7] CUDA libs (if GPU)
:: ----------------------------------------------------------------
echo.
if not "!INSTALL_CUDNN!"=="1" (
    echo [7/9] Skipping CUDA libs ^(CPU mode^).
    goto AFTER_CUDA
)
echo [7/9] Installing CUDA libraries...
"%VENV_PY%" -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
if errorlevel 1 (
    echo [ERROR] CUDA library installation failed.
    pause
    exit /b 1
)
echo [OK] CUDA libs installed.
:AFTER_CUDA

:: ----------------------------------------------------------------
:: [8] mamba_ssm shim for RE-USE
:: ----------------------------------------------------------------
echo.
echo [8/9] Installing mamba_ssm pure-PyTorch shim for RE-USE...
set "SHIM_SRC=%~dp0mamba_ssm_shim.py"
set "SHIM_DIR=%~dp0%VENV_DIR%\Lib\site-packages\mamba_ssm"

if not exist "!SHIM_SRC!" (
    echo [ERROR] mamba_ssm_shim.py not found next to install.bat
    pause
    exit /b 1
)

if not exist "!SHIM_DIR!" mkdir "!SHIM_DIR!"
copy /Y "!SHIM_SRC!" "!SHIM_DIR!\__init__.py" >nul
echo [OK] mamba_ssm shim installed.

:: ----------------------------------------------------------------
:: [9] Final checks
:: ----------------------------------------------------------------
echo.
echo [9/9] Final checks...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found in system PATH^^!
) else (
    echo [OK] FFmpeg detected.
)
echo.
echo ============================================================
echo INSTALLATION COMPLETE^^! ^(!TORCH_VARIANT! + RE-USE^)
echo ============================================================
echo.
echo You can now run the application.
pause