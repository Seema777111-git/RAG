@echo off
rem Creates .venv, installs dependencies and creates .env (Windows). Safe to re-run.
setlocal
cd /d "%~dp0\.."

if not exist .venv (
  py -3.11 -m venv .venv 2>nul
  if errorlevel 1 python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Could not create .venv. Install Python 3.11+ and try again.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if "%TORCH_CPU%"=="1" python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
if not exist .env copy .env.example .env

echo.
echo Setup complete. Put your key in .env (ANTHROPIC_API_KEY=...), then run:
echo   scripts\run_api.bat   (terminal 1)
echo   scripts\run_ui.bat    (terminal 2)
