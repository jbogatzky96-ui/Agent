@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python wurde nicht gefunden. Bitte Python 3.12 von https://www.python.org/downloads/ installieren
  echo und beim Installieren "Add python.exe to PATH" anhaken.
  pause
  exit /b 1
)

if not exist .venv (
  echo Erstelle virtuelle Umgebung ...
  py -3 -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo Installiere Abhaengigkeiten ...
pip install -r requirements.txt || goto :error

if not exist .env copy .env.example .env >nul
if not exist config.toml copy config.example.toml config.toml >nul
if not exist context xcopy /E /I /Q context.example context >nul

echo.
echo Fertig. Naechste Schritte:
echo   1. .env oeffnen und ANTHROPIC_API_KEY und DEEPGRAM_API_KEY eintragen
echo   2. context\briefing.md fuer deinen Call ausfuellen
echo   3. Test ohne Audio:  run.bat --simulate examples\simulation.txt --context context.example --no-ui
echo   4. Live:             run.bat
pause
exit /b 0

:error
echo Setup fehlgeschlagen, siehe Meldung oben.
pause
exit /b 1
