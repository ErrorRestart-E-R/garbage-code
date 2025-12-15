@echo off
setlocal

REM Run GameHub (FastAPI) as a separate process.
REM - Place this file under AiVutber/
REM - Double-click or run from terminal.

REM Ensure we run from this script directory (AiVutber/)
cd /d "%~dp0"

REM Optional: use UTF-8 code page for Korean logs
chcp 65001 >nul

set HOST=127.0.0.1
set PORT=8765

echo [GameHub] Starting on http://%HOST%:%PORT%
echo [GameHub] Press Ctrl+C to stop.
echo.

python -m uvicorn game_hub.server:app --host %HOST% --port %PORT%

echo.
echo [GameHub] Stopped.
pause


