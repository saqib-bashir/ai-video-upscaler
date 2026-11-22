@echo off
REM Drag one or more videos (or a folder) onto this file.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
aiupscale upscale %* --target 4k
pause
