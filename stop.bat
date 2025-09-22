@echo off
echo Beende KI-NPC Fenster...
for /f "tokens=2 delims=," %%i in ('tasklist /v /fo csv ^| findstr /i /c:"KI-NPC App" /c:"npc_bot" /c:"npc_bot_proximity"') do (
  echo Schliesse PID %%i
  taskkill /pid %%i /t /f >nul 2>&1
)

REM Zusätzlich: uvicorn auf Port 8002 hart beenden (falls noch offen)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8002 .*LISTENING"') do taskkill /pid %%p /t /f >nul 2>&1

echo Fertig.
