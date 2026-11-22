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
REM CUDA 12.8 build of PyTorch (RTX 20/30/40/50 series). torch AND torchvision must come
REM from this index together: a torchvision from PyPI would pull in PyPI's torch, which
REM is CPU-only on Windows.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
pip install -e . || goto :fail
python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"
if errorlevel 1 (
  echo [!] PyTorch was replaced by the CPU-only build - reinstalling the CUDA build ...
  pip uninstall -y torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
)
aiupscale models --download general general-wdn x2plus || goto :fail
echo.
aiupscale doctor
echo.
echo Done. Double-click "AI Upscaler.bat" to open the window,
echo or drag a video onto upscale_4k.bat / upscale_8k.bat
pause
exit /b 0
:fail
echo [!] Install failed - see the messages above.
pause
exit /b 1
