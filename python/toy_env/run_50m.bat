@echo off
cd /d E:\Repos\source-sdk-2013\python\toy_env
set PY=E:\Repos\source-sdk-2013\python\.venv\Scripts\python.exe
for /l %%i in (1,1,60) do (
  findstr /c:"done, final model saved" "%1" >nul && goto :done
  echo [wrapper] starting/resuming from latest snapshot, attempt %%i >> "%1"
  "%PY%" -u resume_train.py >> "%1" 2>&1
)
:done
echo [wrapper] finished >> "%1"
