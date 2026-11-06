from contextlib import nullcontext
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from ai_interviewer import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pg_home, self.pg_data = self.root / "pgsql", self.root / "data"
        (self.pg_home / "bin").mkdir(parents=True)
        self.pg_data.mkdir()
        for name in ["pg_ctl.exe", "pg_isready.exe"]:
            (self.pg_home / "bin" / name).touch()

    def test_start_checks_readiness_after_postgres_start(self):
        with patch.object(launcher, "_run_pg_command", side_effect=[2, 3, 0, 0]) as run:
            launcher.ensure_postgres(self.pg_home, self.pg_data)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("pg_isready.exe", commands[0][0])
        self.assertEqual(commands[1][1], "status")
        self.assertEqual(commands[2][1], "start")
        self.assertIn("pg_isready.exe", commands[3][0])

    def test_browser_opens_after_ready_and_exit_stops_backend_then_database(self):
        events = []
        process = Mock()
        process.wait.side_effect = lambda: events.append("web_exited") or 0
        with (
            patch.object(launcher, "_backend_ready", return_value=False),
            patch.object(launcher, "ensure_postgres", side_effect=lambda *a: events.append("db_ready")),
            patch.object(launcher.subprocess, "Popen", side_effect=lambda *a, **k: events.append("web_started") or process),
            patch.object(launcher, "_wait_for_backend", side_effect=lambda *a: events.append("web_ready")),
            patch.object(launcher, "_open_browser", side_effect=lambda *a: events.append("browser_opened")),
            patch.object(launcher, "_stop_backend", side_effect=lambda *a: events.append("web_stopped")),
            patch.object(launcher, "stop_postgres", side_effect=lambda *a: events.append("db_stopped")),
        ):
            self.assertEqual(launcher.run_app(pg_home=self.pg_home, pg_data=self.pg_data), 0)
        self.assertEqual(events, ["db_ready", "web_started", "web_ready", "browser_opened", "web_exited", "web_stopped", "db_stopped"])

    def test_startup_failure_and_ctrl_c_clean_up_owned_database(self):
        for error in [TimeoutError("not ready"), KeyboardInterrupt()]:
            with self.subTest(error=type(error).__name__):
                process = Mock(returncode=None)
                with (
                    patch.object(launcher, "_backend_ready", return_value=False),
                    patch.object(launcher, "ensure_postgres"),
                    patch.object(launcher.subprocess, "Popen", return_value=process),
                    patch.object(launcher, "_wait_for_backend", side_effect=error),
                    patch.object(launcher, "_open_browser") as browser,
                    patch.object(launcher, "_stop_backend") as stop_web,
                    patch.object(launcher, "stop_postgres") as stop_db,
                ):
                    self.assertEqual(launcher.run_app(pg_home=self.pg_home, pg_data=self.pg_data), 0 if isinstance(error, KeyboardInterrupt) else 1)
                    browser.assert_not_called()
                    stop_web.assert_called_once_with(process)
                    stop_db.assert_called_once_with(self.pg_home, self.pg_data)

    def test_duplicate_launcher_does_not_take_ownership_of_database(self):
        with (
            patch.object(launcher, "_launcher_lock", return_value=nullcontext(False)),
            patch.object(launcher, "_backend_ready", return_value=True),
            patch.object(launcher, "_open_browser") as browser,
            patch.object(launcher, "run_app") as run,
            patch.object(launcher, "stop_postgres") as stop,
        ):
            self.assertEqual(launcher.main(), 0)
            browser.assert_called_once()
            run.assert_not_called()
            stop.assert_not_called()

    def test_launcher_lock_excludes_second_owner_and_releases_on_exit(self):
        with patch.object(launcher, "__file__", str(self.root / "ai_interviewer" / "launcher.py")):
            with launcher._launcher_lock() as first:
                self.assertTrue(first)
                with launcher._launcher_lock() as second:
                    self.assertFalse(second)
            with launcher._launcher_lock() as third:
                self.assertTrue(third)

    def test_postgres_does_not_inherit_console_ctrl_c(self):
        with patch.object(launcher.subprocess, "run", return_value=Mock(returncode=0)) as run:
            launcher._run_pg_command(["pg_ctl.exe", "status"])
        if launcher.os.name == "nt":
            flags = run.call_args.kwargs["creationflags"]
            self.assertTrue(flags & subprocess.CREATE_NEW_PROCESS_GROUP)
            self.assertTrue(flags & subprocess.CREATE_NO_WINDOW)


if __name__ == "__main__":
    unittest.main()
