@echo off
REM Drag a video onto this file: writes <video>_compare.png (bicubic vs AI, 100% zoom).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
aiupscale compare "%~1" --target 4k -o "%~dpn1_compare.png"
pause
