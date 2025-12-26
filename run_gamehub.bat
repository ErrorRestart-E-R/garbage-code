@echo off
setlocal

REM Run GameHub (FastAPI) as a separate process.
REM - Place this file under AiVutber/
REM - Double-click or run from terminal.

REM Ensure we run from this script directory (AiVutber/)
cd /d "%~dp0"

REM Optional: use UTF-8 code page for Korean logs
chcp 65001 >nul

set "HOST=127.0.0.1"
set "PORT=8765"

REM Use project's virtual environment (.venv) Python
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [GameHub] ERROR: 가상환경 파이썬을 찾을 수 없습니다.
  echo [GameHub] expected: "%VENV_PY%"
  echo [GameHub] 해결: 프로젝트 루트(AiVutber^)에서 ".venv"를 만들고 패키지를 설치하세요.
  echo [GameHub] 예: python -m venv .venv  ^&^& .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)

echo [GameHub] Starting on http://%HOST%:%PORT%
echo [GameHub] Press Ctrl+C to stop.
echo.

"%VENV_PY%" -m uvicorn game_hub.server:app --host %HOST% --port %PORT%

echo.
echo [GameHub] Stopped.
pause


