@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m call_assistant %*
if errorlevel 1 pause
