@echo off
cd /d "%~dp0"
echo Starting Soybrary...
echo.

set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

for /f %%i in ('%PYTHON% -c "import json;c=json.load(open('config.json'));print(c.get('port',8000))"') do set PORT=%%i
if "%PORT%"=="" set PORT=8000

start http://localhost:%PORT%
%PYTHON% server.py
pause
