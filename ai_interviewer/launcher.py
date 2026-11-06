"""启动后端，确认服务可用后再打开浏览器。"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser


def _backend_ready(health_url: str, timeout: float = 1.0) -> bool:
    try:
        with build_opener(ProxyHandler({})).open(health_url, timeout=timeout) as response:
            return response.status == 200 and json.load(response) == {"status": "ok"}
    except (URLError, OSError, ValueError):
        return False


def _wait_for_backend(process: subprocess.Popen, health_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Web application did not become ready within {timeout:g} seconds.")
        if _backend_ready(health_url, timeout=min(1.0, remaining)) and process.poll() is None:
            return
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"Web application exited before it was ready (exit code {process.returncode}).")


def _run_pg_command(command: list[str], timeout: float = 5.0) -> int:
    # PostgreSQL 使用独立的进程组且不附着控制台，避免网页窗口的 Ctrl+C 终止数据库。
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    return subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, timeout=timeout,
    ).returncode


def ensure_postgres(pg_home: Path, pg_data: Path, *, port: int = 5432) -> None:
    ctl = pg_home / "bin" / "pg_ctl.exe"
    ready = pg_home / "bin" / "pg_isready.exe"
    if not ctl.is_file():
        print(f"[WARN] {ctl} not found; PostgreSQL auto-start skipped.", flush=True)
        return
    if not ready.is_file() or not pg_data.is_dir():
        raise RuntimeError("PostgreSQL tools or data directory are missing; check PG_HOME and PG_DATA.")
    probe = [str(ready), "-h", "127.0.0.1", "-p", str(port), "-t", "1", "-q"]
    if _run_pg_command(probe) == 0:
        print("PostgreSQL is ready.", flush=True)
        return

    deadline = time.monotonic() + 30
    status = _run_pg_command([str(ctl), "status", "-D", str(pg_data)])
    if status == 3:
        # 由 PostgreSQL 自行检查 PID 文件，不能仅凭端口占用判断就绪或删除 PID 文件。
        log_dir = pg_data / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"launcher-{time.time_ns()}.log"
        print(f"Starting PostgreSQL; startup log: {log_path}", flush=True)
        code = _run_pg_command(
            [str(ctl), "start", "-D", str(pg_data), "-l", str(log_path), "-w", "-t", "30"],
            timeout=35,
        )
        if code != 0:
            raise RuntimeError(f"PostgreSQL failed to start (exit code {code}). Check {log_path}")
    elif status != 0:
        raise RuntimeError(f"Cannot check PostgreSQL status (exit code {status}); check PG_DATA.")

    while True:
        if _run_pg_command(probe) == 0:
            print("PostgreSQL is ready.", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("PostgreSQL is not accepting connections; check its server logs.")
        time.sleep(0.5)


def _open_browser(url: str) -> None:
    try:
        if not webbrowser.open(url):
            print(f"[WARN] Could not open the browser. Open {url} manually.", flush=True)
    except (webbrowser.Error, OSError) as exc:
        print(f"[WARN] Could not open the browser: {exc}. Open {url} manually.", flush=True)


def stop_postgres(pg_home: Path, pg_data: Path) -> None:
    ctl = str(pg_home / "bin" / "pg_ctl.exe")
    if _run_pg_command([ctl, "status", "-D", str(pg_data)]) == 3:
        return
    print("Stopping PostgreSQL...", flush=True)
    code = _run_pg_command(
        [ctl, "stop", "-D", str(pg_data), "-m", "fast", "-w", "-t", "15"], timeout=20,
    )
    if code != 0:
        raise RuntimeError(f"PostgreSQL did not stop cleanly (exit code {code}); check its server logs.")


@contextmanager
def _launcher_lock():
    """同一项目只有一个启动器负责启停，重复双击不会误停正在使用的数据库。"""
    path = Path(__file__).resolve().parent.parent / ".launcher.lock"
    with path.open("a+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stop_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_app(
    *, port: int = 8000, startup_timeout: float = 60.0,
    pg_home: Path | None = None, pg_data: Path | None = None,
) -> int:
    browser_url = f"http://localhost:{port}/"
    health_url = f"http://127.0.0.1:{port}/api/health"
    process = None
    manage_postgres = False
    try:
        if _backend_ready(health_url):
            print(f"Web application is already running: {browser_url}", flush=True)
            _open_browser(browser_url)
            return 0
        if pg_home is not None and pg_data is not None:
            manage_postgres = (pg_home / "bin" / "pg_ctl.exe").is_file() and pg_data.is_dir()
            ensure_postgres(pg_home, pg_data)
        # 继承当前控制台的输入输出，启动错误和运行日志仍直接可见。
        process = subprocess.Popen(
            [sys.executable, "-m", "ai_interviewer.app", str(port)],
            cwd=Path(__file__).resolve().parent.parent,
        )
        print(f"Waiting for the web application (up to {startup_timeout:g} seconds)...", flush=True)
        _wait_for_backend(process, health_url, startup_timeout)
        print(f"Web application is ready: {browser_url}", flush=True)
        _open_browser(browser_url)
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping the web application...", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return process.returncode if process is not None and process.returncode else 1
    finally:
        try:
            if process is not None:
                _stop_backend(process)
        finally:
            if manage_postgres:
                stop_postgres(pg_home, pg_data)


def main() -> int:
    try:
        with _launcher_lock() as acquired:
            if not acquired:
                if _backend_ready("http://127.0.0.1:8000/api/health"):
                    _open_browser("http://localhost:8000/")
                else:
                    print("Another launcher is starting or stopping the application; please wait.", flush=True)
                return 0
            pg_home = os.environ.get("PG_HOME")
            pg_data = os.environ.get("PG_DATA")
            return run_app(
                pg_home=Path(pg_home) if pg_home else None,
                pg_data=Path(pg_data) if pg_data else None,
            )
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
