@echo off
echo Starting sec-rag Web UI Backend...
cd /d "%~dp0\backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
