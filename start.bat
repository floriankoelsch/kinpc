@echo off
setlocal ENABLEDELAYEDEXPANSION
title KI-NPC Launcher (robust)

REM =========================
REM == Debug / Verhalten ====
REM =========================
set "DEBUG_MODE=1"                  REM 1 = Fenster bleibt IMMER offen + extra Ausgaben
set "ALWAYS_PAUSE=1"                REM 1 = am Ende immer Pause
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\launcher.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo. > "%LOG_FILE%"

call :log "===== Launcher gestartet %DATE% %TIME% ====="

REM =========================
REM == Konfiguration ========
REM =========================
set "APP_FILE=app.py"
set "BOT_FILE=npc_bot_proximity.py"
set "ALT_BOT_FILE=npc_bot.py"

set "APP_MODULE=app:app"
set "APP_HOST=127.0.0.1"
set "APP_PORT=8002"
set "USE_RELOAD=true"

REM =========================
REM == Grund-Setup ==========
REM =========================
cd /d "%~dp0"
call :log "Arbeitsordner: %CD%"

REM --- Sanity Checks ---
if not exist "%APP_FILE%" (
  call :err "[FEHLER] %APP_FILE% nicht gefunden in: %CD%"
  goto :end_fail
)

REM --- Python finden ---
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe"  set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    for /f "delims=" %%P in ('where python') do (
      set "PYTHON_EXE=%%P"
      goto :py_found
    )
  ) else (
    call :err "[FEHLER] Python nicht gefunden. Bitte Python installieren oder venv bereitstellen."
    goto :end_fail
  )
)
:py_found
for /f "delims=" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PYVER=%%v"
echo Verwende: %PYTHON_EXE%  (%PYVER%)
call :log "Python: %PYTHON_EXE%  (%PYVER%)"

REM --- Virtuelle Umgebung sicherstellen ---
if not exist "venv\Scripts\python.exe" (
  call :log "Erstelle venv..."
  "%PYTHON_EXE%" -m venv venv >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    call :err "[FEHLER] venv konnte nicht erstellt werden (Details siehe %LOG_FILE%)."
    goto :end_fail
  )
)
set "VENV_PY=venv\Scripts\python.exe"
set "VENV_PIP=venv\Scripts\pip.exe"

REM --- Abhängigkeiten ---
if exist "requirements.txt" (
  call :log "Installiere requirements.txt…"
  "%VENV_PIP%" install --upgrade pip >> "%LOG_FILE%" 2>&1
  "%VENV_PIP%" install -r requirements.txt >> "%LOG_FILE%" 2>&1
) else (
  call :log "Keine requirements.txt – installiere Mindestpakete…"
  "%VENV_PIP%" install --upgrade pip >> "%LOG_FILE%" 2>&1
  "%VENV_PIP%" install fastapi uvicorn >> "%LOG_FILE%" 2>&1
)

REM --- .env prüfen (nur Hinweis, nicht abbrechen) ---
if not exist ".env" (
  call :warn "[HINWEIS] .env nicht gefunden. API-Keys evtl. fehlen."
)

REM --- Ports checken (nur Hinweis) ---
call :check_port %APP_PORT% "App"

REM =========================
REM == Start: App ===========
REM =========================
set "RELOAD_FLAG="
if /I "%USE_RELOAD%"=="true" set "RELOAD_FLAG=--reload"

call :log "Starte Uvicorn: %APP_MODULE% auf %APP_HOST%:%APP_PORT% ..."
echo Starte Mini-Map (FastAPI) auf %APP_HOST%:%APP_PORT% ...
start "KI-NPC App %APP_PORT%" cmd /k "venv\Scripts\python.exe -m uvicorn %APP_MODULE% --host %APP_HOST% --port %APP_PORT% %RELOAD_FLAG%"

REM kurze Wartezeit
timeout /t 2 /nobreak >nul

REM =========================
REM == Start: Bot ===========
REM =========================
if exist "%BOT_FILE%" (
  call :log "Starte Bot: %BOT_FILE%"
  start "KI-NPC Bot" cmd /k "venv\Scripts\python.exe %BOT_FILE%"
) else (
  if exist "%ALT_BOT_FILE%" (
    call :warn "[HINWEIS] %BOT_FILE% fehlt – starte %ALT_BOT_FILE%."
    start "KI-NPC Bot (alt)" cmd /k "venv\Scripts\python.exe %ALT_BOT_FILE%"
  ) else (
    call :warn "[HINWEIS] Kein Bot-Skript gefunden (%BOT_FILE% / %ALT_BOT_FILE%)."
  )
)

echo.
echo ============================================
echo Alles gestartet (Fenster bleiben offen).
echo Schau in die einzelnen Konsolen fuer Logs.
echo Logdatei: %LOG_FILE%
echo ============================================

goto :end_ok

REM =========================
REM == Hilfsfunktionen ======
REM =========================
:check_port
REM %1=PORT  %2=LABEL
set "PIDINUSE="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%1 .*LISTENING"') do set "PIDINUSE=%%p"
if defined PIDINUSE (
  call :warn "[WARNUNG] Port %1 (%2) wird bereits von PID %PIDINUSE% genutzt."
  set "PIDINUSE="
)
exit /b 0

:log
echo [LOG] %~1
echo [LOG] %~1>> "%LOG_FILE%"
exit /b 0

:warn
echo [WARNUNG] %~1
echo [WARNUNG] %~1>> "%LOG_FILE%"
exit /b 0

:err
echo [FEHLER] %~1
echo [FEHLER] %~1>> "%LOG_FILE%"
exit /b 0

:end_fail
echo.
echo ===== FEHLER =====
echo Letzte 40 Zeilen aus %LOG_FILE%:
echo ------------------------------
powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Tail 40"
echo ------------------------------
if "%ALWAYS_PAUSE%"=="1" pause
exit /b 1

:end_ok
if "%DEBUG_MODE%"=="1" (
  echo.
  echo [DEBUG] Launcher fertig. Fenster bleibt offen.
)
if "%ALWAYS_PAUSE%"=="1" pause
endlocal
exit /b 0
