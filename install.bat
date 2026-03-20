@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title WhisperX Subtitle Generator - Installer
color 0A

echo.
echo  ============================================================
echo    Subtitle Generator - Installer
echo  ============================================================
echo.

:: ----------------------------------------------------------------
:: [1] Check Python
:: ----------------------------------------------------------------
echo [1/8] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found!
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo  Found: %%i

:: ----------------------------------------------------------------
:: [2] GPU Detection
:: ----------------------------------------------------------------
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
    echo  NVIDIA GPU detected.
    echo  Driver major version: !DRIVER_MAJOR!
    echo  Recommended: GPU ^(CUDA^)
) else (
    echo  No NVIDIA GPU detected.
    echo  Recommended: CPU
)
echo.
echo  Choose installation type:
echo.
echo   [1] CPU  (works everywhere^)
echo   [2] GPU  (NVIDIA CUDA - faster transcription^)
echo.
:CHOICE
set USER_CHOICE=
set /p USER_CHOICE= Enter choice [1/2]: 
if "%USER_CHOICE%"=="1" goto CPU_MODE
if "%USER_CHOICE%"=="2" goto GPU_MODE
echo  Invalid choice. Please enter 1 or 2.
goto CHOICE

:: ----------------------------------------------------------------
:: CPU MODE
:: ----------------------------------------------------------------
:CPU_MODE
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
set INSTALL_CUDNN=0
goto CREATE_VENV

:: ----------------------------------------------------------------
:: GPU MODE
:: ----------------------------------------------------------------
:GPU_MODE
if !GPU_FOUND! EQU 0 (
    echo.
    echo  WARNING: No NVIDIA GPU detected.
    echo  Installing CPU version instead.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback ^(no GPU found^)
    set INSTALL_CUDNN=0
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    set INSTALL_CUDNN=1
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    set INSTALL_CUDNN=1
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    set INSTALL_CUDNN=1
    goto CREATE_VENV
)
echo  Driver version !DRIVER_MAJOR! is too old (minimum 450^).
echo  Installing CPU version instead.
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU fallback ^(driver too old^)
set INSTALL_CUDNN=0

:: ----------------------------------------------------------------
:: [3] Create virtual environment
:: ----------------------------------------------------------------
:CREATE_VENV
echo.
echo  Selected mode: !TORCH_VARIANT!
echo.
echo [3/8] Creating virtual environment (.venv^)...
if exist ".venv" (
    echo  .venv already exists - skipping creation.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  .venv created OK.
)

:: ----------------------------------------------------------------
:: [4] Activate venv and upgrade pip
:: ----------------------------------------------------------------
echo.
echo [4/8] Activating .venv and upgrading pip...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
python -m pip install --upgrade pip --quiet
echo  pip upgraded OK.

:: ----------------------------------------------------------------
:: [5] Install PyTorch
:: ----------------------------------------------------------------
echo.
echo [5/8] Installing PyTorch ^(!TORCH_VARIANT!^)...
echo  This may take several minutes...
python -m pip install torch torchaudio --index-url !TORCH_INDEX_URL! --quiet
if errorlevel 1 (
    echo  ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)
echo  PyTorch installed OK.
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"

:: ----------------------------------------------------------------
:: [6] Install requirements.txt
:: ----------------------------------------------------------------
echo.
echo [6/8] Installing packages from requirements.txt...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  ERROR: Some packages failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)
echo  All requirements installed OK.

:: ----------------------------------------------------------------
:: [7] Install nvidia-cudnn-cu12 (GPU mode only)
:: ----------------------------------------------------------------
echo.
if !INSTALL_CUDNN! EQU 1 (
    echo [7/8] Installing nvidia-cudnn-cu12 ^(required for CUDA float16 inference^)...
    echo  This may take a few minutes ~720 MB...
    python -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 --quiet
    if errorlevel 1 (
        echo  WARNING: nvidia-cudnn-cu12 installation failed.
        echo  CUDA float16 inference may not work.
    ) else (
        echo  nvidia-cudnn-cu12 installed OK.
    )
) else (
    echo [7/8] Skipping nvidia-cudnn-cu12 ^(CPU mode - not needed^).
)

:: ----------------------------------------------------------------
:: [8] Check FFmpeg + install Demucs
:: ----------------------------------------------------------------
echo.
echo [8/8] Final checks...
echo.
echo  Checking FFmpeg ^(required for video files^)...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo  WARNING: FFmpeg not found in PATH!
    echo  Video files ^(.mp4, .mkv, etc.^) will NOT work without it.
    echo.
    echo  Install options:
    echo    winget install FFmpeg
    echo    or download from: https://www.gyan.dev/ffmpeg/builds/
) else (
    echo  FFmpeg found OK.
)

echo.
echo  Installing Demucs ^(voice separation^)...
python -m pip install demucs --quiet
if errorlevel 1 (
    echo  WARNING: Demucs installation failed. Voice separation will not work.
) else (
    echo  Demucs installed OK.
)

:: ----------------------------------------------------------------
:: Final verification
:: ----------------------------------------------------------------
echo.
echo  ============================================================
echo    Verifying installation...
echo  ============================================================
python -c "import whisperx; import PyQt6; import sounddevice; import pysrt; print('  Core packages verified OK!')"
if errorlevel 1 (
    echo  WARNING: Some packages may be missing. Check errors above.
)
if !INSTALL_CUDNN! EQU 1 (
    python -c "import nvidia.cudnn; print('  nvidia-cudnn-cu12 verified OK!')" 2>nul || echo  WARNING: nvidia-cudnn-cu12 not importable - CUDA float16 may not work.
)

call .venv\Scripts\deactivate.bat 2>nul

echo.
echo  ============================================================
echo    Installation complete! ^(!TORCH_VARIANT!^)
echo    Run the app by double-clicking: launcher.vbs
echo  ============================================================
echo.
pause
