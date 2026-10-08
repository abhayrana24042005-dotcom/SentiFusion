@echo off
echo ===================================================
echo   Starting SentiFusion Multimodal AI Platform
echo ===================================================
echo.
echo 1. Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "SentiFusion Backend (FastAPI)" cmd /k "cd backend && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo 2. Launching React Frontend on http://127.0.0.1:5173 ...
start "SentiFusion Frontend (React)" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo   Application Started!
echo   Open your browser at: http://127.0.0.1:5173
echo ===================================================
echo.
pause
