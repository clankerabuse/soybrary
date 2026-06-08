@echo off
cd /d "%~dp0"
echo Starting Soybrary...
echo.
start http://localhost:8000
.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
