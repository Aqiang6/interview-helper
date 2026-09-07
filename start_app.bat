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
            "%PG_HOME%\bin\pg_ctl.exe" start -D "%PG_DATA%" -l "%PG_DATA%\pg.log"
            if errorlevel 1 (
                echo.
                echo [WARN] PostgreSQL failed to start. Check log: %PG_DATA%\pg.log
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
    echo [1/2] %PG_HOME%\bin\pg_ctl.exe not found, skip PostgreSQL auto-start.
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
set "APP_EXIT=%errorlevel%"

echo.
if "%APP_EXIT%"=="0" (
    echo Web application exited normally.
) else (
    echo [ERROR] Web application exited with code: %APP_EXIT%
    echo        Please check the log above.
)

REM ----- Stop PostgreSQL on exit -----
if exist "%PG_HOME%\bin\pg_ctl.exe" (
    echo.
    echo Stopping PostgreSQL...
    "%PG_HOME%\bin\pg_ctl.exe" stop -D "%PG_DATA%" -m fast
)

echo.
echo Press any key to close...
pause >nul
