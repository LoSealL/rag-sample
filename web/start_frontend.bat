@echo off
echo Starting sec-rag Web UI Frontend...
cd /d "%~dp0\frontend"
npm install
npm run dev
