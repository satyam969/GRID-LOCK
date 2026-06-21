@echo off
echo Starting TrafficGuard AI Servers...

:: Start the FastAPI backend in a new command prompt window
echo Starting Backend...
start "TrafficGuard Backend" cmd /k "cd /d %~dp0backend && ..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

:: Start the React frontend in a new command prompt window
echo Starting Frontend...
start "TrafficGuard Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Both servers are starting in new windows!
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:5173
pause
