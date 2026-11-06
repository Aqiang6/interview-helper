@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Could not open the application directory.
    pause
    exit /b 1
)
echo ========================================
echo   AI Interview - Web Application
echo ========================================
echo.

set "PG_HOME=D:\postgresql\pgsql"
set "PG_DATA=D:\postgresql\data"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Please install Python and add to PATH.
    echo.
    pause
    exit /b 1
)

REM Check PostgreSQL readiness, then start/reuse the backend and open the browser.
REM The launcher stops the web backend and then shuts down PostgreSQL cleanly on exit.
python -m ai_interviewer.launcher
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Web application stopped with exit code %APP_EXIT_CODE%.
    echo         Check the error output above.
    pause
)

exit /b %APP_EXIT_CODE%
