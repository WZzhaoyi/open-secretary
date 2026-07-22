#!/usr/bin/env python3
"""Rebuild the Secretary SQLite database while retaining active state only.

The systemd wrapper in ``rebuild-database.sh`` stops the service before this
module runs. This module creates a verified logical archive, initializes a
database from the current SQLAlchemy models, migrates open events and every
scheduled task, and atomically replaces the live database.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


REPO_DIR = Path(__file__).resolve().parents[1]
SECRETARY_DIR = REPO_DIR / "secretary_v2"
if str(SECRETARY_DIR) not in sys.path:
    sys.path.insert(0, str(SECRETARY_DIR))

from config import get_config  # noqa: E402
from memory import Database  # noqa: E402


EVENT_COLUMNS = (
    "id",
    "type",
    "status",
    "summary",
    "content",
    "created_at",
    "source_channel",
    "session_key",
    "source_message_id",
    "metadata_json",
)
TASK_COLUMNS = (
    "id",
    "cron",
    "prompt",
    "enabled",
    "protected",
    "last_run",
    "created_at",
)
EMPTY_TABLES = ("messages", "agent_events", "subagent_runs")
TIMESTAMP_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}$")


class RebuildError(RuntimeError):
    """Raised when a database cannot be rebuilt safely."""


@dataclass(frozen=True)
class RebuildResult:
    database_path: Path
    archive_path: Path
    open_events: int
    scheduled_tasks: int
    archived_events: int


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def _quick_check(connection: sqlite3.Connection) -> None:
    result = connection.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        raise RebuildError(f"SQLite quick_check failed: {result!r}")
    foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_issues:
        raise RebuildError(
            f"SQLite foreign_key_check found {len(foreign_key_issues)} issue(s)"
        )


def _select_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    where: str = "",
    parameters: Iterable[object] = (),
) -> list[tuple]:
    column_sql = ", ".join(columns)
    where_sql = f" WHERE {where}" if where else ""
    return connection.execute(
        f"SELECT {column_sql} FROM {table}{where_sql} ORDER BY id",
        tuple(parameters),
    ).fetchall()


def _backup_database(source: Path, destination: Path, mode: int) -> None:
    with _connect_readonly(source) as source_connection:
        _quick_check(source_connection)
        with sqlite3.connect(destination) as destination_connection:
            source_connection.backup(destination_connection)
    os.chmod(destination, mode)
    with _connect_readonly(destination) as archived_connection:
        _quick_check(archived_connection)


def _create_fresh_database(path: Path) -> None:
    database = Database(str(path))
    database.engine.dispose()


def _insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[tuple],
) -> None:
    if not rows:
        return
    column_sql = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})", rows
    )


def _populate_fresh_database(archive: Path, fresh: Path) -> tuple[int, int, int]:
    with _connect_readonly(archive) as archived_connection:
        archived_events = archived_connection.execute(
            "SELECT count(*) FROM events"
        ).fetchone()[0]
        open_events = _select_rows(
            archived_connection,
            "events",
            EVENT_COLUMNS,
            "status = ?",
            ("open",),
        )
        scheduled_tasks = _select_rows(
            archived_connection, "scheduled_tasks", TASK_COLUMNS
        )

    with sqlite3.connect(fresh) as fresh_connection:
        with fresh_connection:
            _insert_rows(fresh_connection, "events", EVENT_COLUMNS, open_events)
            _insert_rows(
                fresh_connection, "scheduled_tasks", TASK_COLUMNS, scheduled_tasks
            )
        _quick_check(fresh_connection)
        if _select_rows(fresh_connection, "events", EVENT_COLUMNS) != open_events:
            raise RebuildError("open events differ after migration")
        if (
            _select_rows(fresh_connection, "scheduled_tasks", TASK_COLUMNS)
            != scheduled_tasks
        ):
            raise RebuildError("scheduled tasks differ after migration")
        for table in EMPTY_TABLES:
            count = fresh_connection.execute(
                f"SELECT count(*) FROM {table}"
            ).fetchone()[0]
            if count != 0:
                raise RebuildError(f"fresh table {table} is not empty")

    return len(open_events), len(scheduled_tasks), int(archived_events)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def rebuild_database(
    database_path: Path,
    archive_dir: Path | None = None,
    timestamp: str | None = None,
) -> RebuildResult:
    """Archive and rebuild one stopped Secretary SQLite database."""
    source = database_path.expanduser().resolve()
    if not source.is_file():
        raise RebuildError(f"database does not exist: {source}")

    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise RebuildError("timestamp must use YYYYMMDD-HHMMSS")

    archive_directory = (
        archive_dir.expanduser().resolve()
        if archive_dir is not None
        else source.parent / "archive"
    )
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive = archive_directory / f"{source.stem}-{timestamp}{source.suffix}"
    if archive.exists():
        raise RebuildError(f"archive already exists: {archive}")

    source_mode = stat.S_IMODE(source.stat().st_mode)
    process_id = os.getpid()
    archive_temporary = archive_directory / f".{archive.name}.partial-{process_id}"
    fresh_temporary = source.parent / f".{source.name}.rebuild-{timestamp}-{process_id}"
    switched = False

    if archive_temporary.exists() or fresh_temporary.exists():
        raise RebuildError("temporary rebuild path already exists")

    try:
        _backup_database(source, archive_temporary, source_mode)
        os.replace(archive_temporary, archive)
        _fsync_directory(archive_directory)

        _create_fresh_database(fresh_temporary)
        open_events, scheduled_tasks, archived_events = _populate_fresh_database(
            archive, fresh_temporary
        )
        os.chmod(fresh_temporary, source_mode)

        os.replace(fresh_temporary, source)
        switched = True
        _fsync_directory(source.parent)
        with _connect_readonly(source) as live_connection:
            _quick_check(live_connection)
    except Exception as error:
        _remove_temporary(archive_temporary)
        _remove_temporary(fresh_temporary)
        if switched:
            restore_temporary = source.parent / f".{source.name}.restore-{process_id}"
            try:
                _backup_database(archive, restore_temporary, source_mode)
                os.replace(restore_temporary, source)
                _fsync_directory(source.parent)
            except Exception as restore_error:
                raise RebuildError(
                    f"rebuild failed ({error}); automatic restore also failed "
                    f"({restore_error}); archive is at {archive}"
                ) from restore_error
            raise RebuildError(
                f"rebuild failed after cutover; original database restored from "
                f"{archive}: {error}"
            ) from error
        if isinstance(error, RebuildError):
            raise
        raise RebuildError(f"rebuild failed before cutover: {error}") from error

    return RebuildResult(
        database_path=source,
        archive_path=archive,
        open_events=open_events,
        scheduled_tasks=scheduled_tasks,
        archived_events=archived_events,
    )


def _configured_database_path() -> Path:
    configured = Path(get_config().database.path).expanduser()
    if not configured.is_absolute():
        configured = SECRETARY_DIR / configured
    return configured


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the Secretary SQLite database, recreate its schema, and retain "
            "only open events and scheduled tasks. Stop the service before running."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="database path (default: database.path from secretary_v2/config.yaml)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="archive directory (default: an archive directory beside the database)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    database_path = args.database or _configured_database_path()
    try:
        result = rebuild_database(database_path, args.archive_dir)
    except RebuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Database rebuilt: {result.database_path}")
    print(f"Archive created: {result.archive_path}")
    print(f"Open events migrated: {result.open_events}")
    print(f"Scheduled tasks migrated: {result.scheduled_tasks}")
    print(f"Archived events retained: {result.archived_events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
