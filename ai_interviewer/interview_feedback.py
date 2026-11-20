"""按浏览器保存每轮面试反馈，为后续出题提供近期偏好。"""

import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path

MAX_FEEDBACK_LENGTH = 1000
_PROMPT_FEEDBACK_LIMIT = 10
_DB_PATH = Path(__file__).resolve().parent / ".interview_feedback" / "feedback.sqlite3"


@contextmanager
def _database():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(_DB_PATH)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS interview_feedback (
                browser_id TEXT NOT NULL,
                round_id TEXT NOT NULL,
                feedback TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (browser_id, round_id)
            )
        """)
        yield db


def save_feedback(browser_id: str, round_id: str, feedback: str) -> None:
    with _database() as db:
        db.execute("""
            INSERT INTO interview_feedback (browser_id, round_id, feedback, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (browser_id, round_id) DO UPDATE SET
                feedback = excluded.feedback,
                updated_at = excluded.updated_at
            WHERE interview_feedback.feedback <> excluded.feedback
        """, (browser_id, round_id, feedback, time.time_ns()))


def recent_feedback(browser_id: str) -> list[str]:
    if not _DB_PATH.is_file():
        return []
    with _database() as db:
        rows = db.execute("""
            SELECT feedback FROM interview_feedback
            WHERE browser_id = ?
            GROUP BY feedback
            ORDER BY MAX(updated_at) DESC, MAX(rowid) DESC
            LIMIT ?
        """, (browser_id, _PROMPT_FEEDBACK_LIMIT)).fetchall()
    return [feedback for (feedback,) in rows]
