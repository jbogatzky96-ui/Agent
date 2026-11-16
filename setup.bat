@echo off
setlocal
cd /d "%~dp0"

rem --- Python 3.11 oder neuer finden (auch wenn zusaetzlich eine aeltere Version installiert ist) ---
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  for %%V in (3.12 3.13 3.14 3.11) do (
    if not defined PY (
      py -%%V -c "pass" >nul 2>nul && set "PY=py -%%V"
    )
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo Es wurde kein Python 3.11 oder neuer gefunden.
  echo Auf diesem Rechner ist nur eine aeltere Version installiert, oder gar keine.
  echo.
  echo Bitte Python 3.12 von https://www.python.org/downloads/ installieren
  echo und beim Installieren "Add python.exe to PATH" anhaken.
  echo Die alte Version kann installiert bleiben. Danach setup.bat erneut starten.
  pause
  exit /b 1
)
echo Verwende:
%PY% --version

rem --- Eine virtuelle Umgebung mit zu alter Python-Version (z. B. von einem fehlgeschlagenen Versuch) ersetzen ---
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Vorhandene virtuelle Umgebung nutzt eine zu alte Python-Version und wird neu erstellt ...
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe (
  if exist .venv rmdir /s /q .venv
  echo Erstelle virtuelle Umgebung ...
  %PY% -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo Installiere Abhaengigkeiten ...
python -m pip install -r requirements.txt || goto :error

if not exist .env copy .env.example .env >nul
if not exist config.toml copy config.example.toml config.toml >nul
if not exist context xcopy /E /I /Q context.example context >nul

echo.
echo Fertig. Naechste Schritte:
echo   1. .env oeffnen und ANTHROPIC_API_KEY und DEEPGRAM_API_KEY eintragen
echo   2. Deinen context-Ordner vom alten Rechner hierher kopieren (oder ueber "Kontext" im Fenster neu anlegen)
echo   3. Test ohne Audio:  run.bat --simulate examples\simulation.txt --context context.example --no-ui
echo   4. Live:             run.bat
pause
exit /b 0

:error
echo Setup fehlgeschlagen, siehe Meldung oben.
pause
exit /b 1
