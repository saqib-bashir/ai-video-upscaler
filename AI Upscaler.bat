@echo off
REM Opens the drag-and-drop window. You can also drop videos onto this file.
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m aiupscale.gui %*
