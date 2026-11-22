@echo off
REM Fixes "No GPU found": replaces a CPU-only PyTorch with the CUDA build.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run install.bat first.
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo ==== Current setup ====
aiupscale doctor
echo.
echo ==== Reinstalling PyTorch with CUDA (about 3 GB download) ====
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo [!] Download failed - check your internet connection and run fix_gpu.bat again.
  pause & exit /b 1
)
echo.
echo ==== After the fix ====
aiupscale doctor
if errorlevel 1 (
  echo.
  echo [!] Still no GPU. Read the "cause" line above: usually the laptop's
  echo     Eco / iGPU-only mode is on, or the NVIDIA driver needs updating.
) else (
  echo.
  echo GPU is working. Close this window and reopen "AI Upscaler.bat".
)
pause
