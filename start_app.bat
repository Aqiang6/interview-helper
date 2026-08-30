@echo off
chcp 65001 >nul
echo ========================================
echo   AI 模拟面试 - Web 应用
echo ========================================
echo.

set "PG_HOME=D:\postgresql\pgsql"
set "PG_DATA=D:\postgresql\data"

REM ----- 1. 启动 PostgreSQL -----
if exist "%PG_HOME%\bin\pg_ctl.exe" (
    echo [1/2] 启动 PostgreSQL...
    "%PG_HOME%\bin\pg_ctl.exe" status -D "%PG_DATA%" >nul 2>&1
    if errorlevel 1 (
        "%PG_HOME%\bin\pg_ctl.exe" start -D "%PG_DATA%" -l "%PG_DATA%\pg.log"
        if errorlevel 1 (
            echo.
            echo [警告] PostgreSQL 启动失败，请检查端口或日志: %PG_DATA%\pg.log
            echo        继续尝试启动 Web 应用，数据库接口会报错直到 PG 就绪。
            echo.
        ) else (
            echo       PostgreSQL 已就绪。
        )
    ) else (
        echo       PostgreSQL 已在运行，跳过启动。
    )
) else (
    echo [1/2] 未找到 %PG_HOME%\bin\pg_ctl.exe，跳过 PostgreSQL 自动启动。
)

REM ----- 2. 启动 Web 应用 -----
echo.
echo [2/2] 启动 Web 应用 -> http://localhost:8000
echo.
python -m ai_interviewer.app <NUL 2>NUL

REM ----- 用户按 Ctrl+C 关闭后，可选停掉 PG（保留在后台也可） -----
if exist "%PG_HOME%\bin\pg_ctl.exe" (
    echo.
    echo 正在停止 PostgreSQL...
    "%PG_HOME%\bin\pg_ctl.exe" stop -D "%PG_DATA%" -m fast
)
