@echo off
cd /d E:\Repos\source-sdk-2013\python\toy_env
set PY=E:\Repos\source-sdk-2013\python\.venv\Scripts\python.exe
for /l %%i in (1,1,30) do (
  findstr /c:"[hard] done, saved to" "%1" >nul && goto :done
  echo [wrapper] starting/restarting hard fine-tune, attempt %%i >> "%1"
  "%PY%" -u finetune_hard.py >> "%1" 2>&1
)
:done
echo [wrapper] finished >> "%1"
