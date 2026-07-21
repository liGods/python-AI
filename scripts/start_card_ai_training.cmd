@echo off
setlocal
set "CARD_AI_INFERENCE_BACKEND=cuda"
cd /d "E:\SGBJP"
"E:\SGBJP\.venv\Scripts\python.exe" -m ok_tasks.card_ai continuous --project-root "E:\SGBJP" --target-games 200000 --batch-games 10000 --train-every 200000 --evaluation-deals 50000 --workers 12
exit /b %ERRORLEVEL%
