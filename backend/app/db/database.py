"""SQLite helpers for storing chat history during local sessions."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.app.utils.helpers import SQLITE_PATH


def get_connection(db_path: Path = SQLITE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with rows addressable by column name."""

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Create the chat history table if it does not exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def add_chat_message(question: str, answer: str) -> None:
    """Persist one question and answer pair."""

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_history (question, answer, created_at)
            VALUES (?, ?, ?)
            """,
            (question, answer, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()


def get_chat_history(limit: int = 20) -> list[dict[str, str]]:
    """Return recent chat history, oldest first for display."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT question, answer, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_chat_history() -> None:
    """Delete all saved messages for the current local session."""

    with get_connection() as connection:
        connection.execute("DELETE FROM chat_history")
        connection.commit()
