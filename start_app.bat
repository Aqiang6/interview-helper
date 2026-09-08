@echo off
chcp 65001 >nul
echo ========================================
echo   AI Interview - Web Application
echo ========================================
echo.

set "PG_HOME=D:\postgresql\pgsql"
set "PG_DATA=D:\postgresql\data"

REM ----- 1. Start PostgreSQL -----
if exist "%PG_HOME%\bin\pg_ctl.exe" (
    echo [1/2] Starting PostgreSQL...
    "%PG_HOME%\bin\pg_ctl.exe" status -D "%PG_DATA%" >nul 2>&1
    if errorlevel 1 (
        REM status failed, check if port 5432 is already in use (PG might be running)
        netstat -ano | findstr ":5432" | findstr "LISTENING" >nul 2>&1
        if errorlevel 1 (
            REM PG not running and port free: remove stale postmaster.pid so pg_ctl start won't hang
            if exist "%PG_DATA%\postmaster.pid" (
                echo       Removing stale postmaster.pid...
                del /q "%PG_DATA%\postmaster.pid" 2>nul
            )
            REM Start PG with -l NUL to avoid pg.log file lock issues
            REM (Windows antivirus may briefly lock freshly created pg.log, causing
            REM 30s "sharing violation" retry loop in PG startup process)
            "%PG_HOME%\bin\pg_ctl.exe" start -D "%PG_DATA%" -l NUL
            if errorlevel 1 (
                echo.
                echo [WARN] PostgreSQL failed to start. Check log: %PG_DATA%\log\
                echo        Will continue to start web app. DB calls will fail until PG is ready.
                echo.
            ) else (
                echo       PostgreSQL is ready.
            )
        ) else (
            echo       Port 5432 is in use, PostgreSQL seems already running, skip start.
        )
    ) else (
        echo       PostgreSQL is already running, skip start.
    )
) else (
    echo [1/1] %PG_HOME%\bin\pg_ctl.exe not found, skip PostgreSQL auto-start.
)

REM ----- 2. Start Web Application -----
echo.
echo [2/2] Starting Web Application -^> http://localhost:8000
echo.

REM Kill any leftover process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo       Killing leftover process on port 8000 (PID %%a^)...
    taskkill /F /PID %%a >nul 2>&1
)
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Please install Python and add to PATH.
    echo.
    pause
    exit /b 1
)

python -m ai_interviewer.app

REM ----- Force kill any leftover python on port 8000 (uvicorn may not exit on first Ctrl+C) -----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM ----- Stop PostgreSQL on exit (suppress errors to avoid extra Ctrl+C prompts) -----
if exist "%PG_HOME%\bin\pg_ctl.exe" if exist "%PG_DATA%\postmaster.pid" (
    echo Stopping PostgreSQL...
    "%PG_HOME%\bin\pg_ctl.exe" stop -D "%PG_DATA%" -m fast 2>nul
)

exit /b 0
