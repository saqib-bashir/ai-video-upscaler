@echo off
REM One-time setup for Windows + NVIDIA GPU (e.g. RTX 4070 Laptop).
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [!] Python not found. Install Python 3.10 - 3.13 from https://www.python.org/downloads/
  echo     and tick "Add python.exe to PATH". Then run install.bat again.
  pause & exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [*] ffmpeg not found - installing with winget ...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  echo [!] Close this window and run install.bat again so the new PATH is picked up.
  pause & exit /b 1
)

if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
REM CUDA 12.8 build of PyTorch (supports RTX 20/30/40/50 series)
pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :fail
pip install -e . || goto :fail
aiupscale models --download general general-wdn x2plus || goto :fail
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA OK:', ok, torch.cuda.get_device_name(0) if ok else '- running on CPU will be VERY slow')"
echo.
echo Done. Double-click "AI Upscaler.bat" to open the window,
echo or drag a video onto upscale_4k.bat / upscale_8k.bat
pause
exit /b 0
:fail
echo [!] Install failed - see the messages above.
pause
exit /b 1
