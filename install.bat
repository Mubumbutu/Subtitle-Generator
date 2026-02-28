@echo off
chcp 65001 >nul
title WhisperX Subtitle Generator - Installer
color 0A

echo.
echo  ============================================================
echo    Subtitle Generator - Installer
echo  ============================================================
echo.

:: ----------------------------------------------------------------
:: [1] Sprawdz Python
:: ----------------------------------------------------------------
echo [1/7] Checking Python...
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
:: [2] Utworz wirtualne srodowisko
:: ----------------------------------------------------------------
echo.
echo [2/7] Creating virtual environment (.venv)...
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
:: [3] Aktywuj venv i uaktualnij pip
:: ----------------------------------------------------------------
echo.
echo [3/7] Activating .venv and upgrading pip...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
python -m pip install --upgrade pip --quiet
echo  pip upgraded OK.

:: ----------------------------------------------------------------
:: [4] Zainstaluj PyTorch (CUDA 12.1)
:: ----------------------------------------------------------------
echo.
echo [4/7] Installing PyTorch with CUDA 12.1 support...
echo  This may take several minutes...
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo  WARNING: CUDA build failed, falling back to CPU-only PyTorch...
    python -m pip install torch torchaudio --quiet
    if errorlevel 1 (
        echo  ERROR: Failed to install PyTorch.
        pause
        exit /b 1
    )
    echo  CPU-only PyTorch installed.
)
echo  PyTorch installed OK.
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"

:: ----------------------------------------------------------------
:: [5] Zainstaluj requirements.txt
:: ----------------------------------------------------------------
echo.
echo [5/7] Installing packages from requirements.txt...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  ERROR: Some packages failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)
echo  All requirements installed OK.

:: ----------------------------------------------------------------
:: [6] Sprawdz FFmpeg
:: ----------------------------------------------------------------
echo.
echo [6/7] Checking FFmpeg (required for video files^)...
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

:: ----------------------------------------------------------------
:: [7] Zainstaluj Demucs
:: ----------------------------------------------------------------
echo.
echo [7/7] Installing Demucs (voice separation^)...
python -m pip install demucs --quiet
if errorlevel 1 (
    echo  WARNING: Demucs installation failed. Voice separation will not work.
) else (
    echo  Demucs installed OK.
)

:: ----------------------------------------------------------------
:: Weryfikacja koncowa
:: ----------------------------------------------------------------
echo.
echo  ============================================================
echo    Verifying installation...
echo  ============================================================
python -c "import whisperx; import PyQt6; import sounddevice; import pysrt; print('  Core packages verified OK!')"
if errorlevel 1 (
    echo  WARNING: Some packages may be missing. Check errors above.
)

call .venv\Scripts\deactivate.bat 2>nul

echo.
echo  ============================================================
echo    Installation complete!
echo    Run the app by double-clicking: launcher.vbs
echo  ============================================================
echo.
pause
