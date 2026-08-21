"""Tests for the deploy-time SQLite rebuild utility."""

import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

from memory import Database


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "deploy" / "rebuild_database.py"
SPEC = importlib.util.spec_from_file_location("rebuild_database", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
rebuild_database = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rebuild_database
SPEC.loader.exec_module(rebuild_database)


def _rows(database_path, sql, parameters=()):
    with sqlite3.connect(database_path) as connection:
        return connection.execute(sql, parameters).fetchall()


def test_rebuild_retains_only_open_events_and_scheduled_tasks(tmp_path):
    database_path = tmp_path / "secretary.db"
    archive_dir = tmp_path / "archive"
    database = Database(str(database_path))

    first_open = database.create_event(
        "remind",
        "follow this",
        status="open",
        summary="first open",
        source_channel="feishu",
        session_key="feishu:chat:sender",
        source_message_id="message-1",
        metadata={"key": "value"},
    )
    database.create_event("note", "historical", status="logged")
    second_open = database.create_event(
        "check", "check later", status="open", summary="second open"
    )
    database.create_event("response", "finished", status="resolved")
    database.save_message(source="user", content="old conversation")
    database.create_scheduled_task("enabled", "0 8 * * *", "enabled prompt")
    database.create_scheduled_task(
        "disabled",
        "30 9 * * 1-5",
        "",
        protected=True,
        handler="builtin",
        builtin_task="maintenance",
    )
    database.update_scheduled_task(
        "disabled",
        enabled=0,
        last_run=datetime(2026, 7, 20, 1, 2, 3),
        last_attempt=datetime(2026, 7, 20, 1, 0, 0),
        last_success=datetime(2026, 7, 20, 1, 2, 3),
    )
    database.engine.dispose()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO agent_events (origin, type, subject) VALUES (?, ?, ?)",
            ("test", "run_finished", "old audit"),
        )
        connection.execute(
            "INSERT INTO subagent_runs "
            "(id, agent_name, agent_kind, engine, input_json, status, origin_channel) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("run-1", "research", "test", "agent", "{}", "done", "cli"),
        )
        connection.commit()

    result = rebuild_database.rebuild_database(
        database_path,
        archive_dir=archive_dir,
        timestamp="20260722-120000",
    )

    assert result.database_path == database_path.resolve()
    assert result.archive_path == archive_dir / "secretary-20260722-120000.db"
    assert result.open_events == 2
    assert result.scheduled_tasks == 2
    assert result.archived_events == 4

    live_events = _rows(
        database_path,
        "SELECT id, status, summary FROM events ORDER BY id",
    )
    assert live_events == [
        (first_open.id, "open", "first open"),
        (second_open.id, "open", "second open"),
    ]
    assert _rows(
        database_path,
        "SELECT id, enabled, protected, handler, builtin_task, last_run, "
        "last_attempt, last_success, last_error "
        "FROM scheduled_tasks ORDER BY id",
    ) == [
        (
            "disabled", 0, 1, "builtin", "maintenance",
            "2026-07-20 01:02:03.000000",
            "2026-07-20 01:00:00.000000",
            "2026-07-20 01:02:03.000000", None,
        ),
        ("enabled", 1, 0, "agent", None, None, None, None, None),
    ]
    for table in ("messages", "agent_events", "subagent_runs"):
        assert _rows(database_path, f"SELECT count(*) FROM {table}") == [(0,)]

    archive_path = result.archive_path
    assert _rows(archive_path, "SELECT count(*) FROM events") == [(4,)]
    assert _rows(archive_path, "SELECT count(*) FROM messages") == [(1,)]
    assert _rows(archive_path, "SELECT count(*) FROM agent_events") == [(1,)]
    assert _rows(archive_path, "SELECT count(*) FROM subagent_runs") == [(1,)]

def test_rebuild_refuses_to_overwrite_existing_archive(tmp_path):
    database_path = tmp_path / "secretary.db"
    archive_dir = tmp_path / "archive"
    database = Database(str(database_path))
    database.create_event("note", "keep me", status="logged")
    database.engine.dispose()

    archive_dir.mkdir()
    archive_path = archive_dir / "secretary-20260722-120000.db"
    archive_path.write_bytes(b"existing archive")
    original = database_path.read_bytes()

    with pytest.raises(rebuild_database.RebuildError, match="archive already exists"):
        rebuild_database.rebuild_database(
            database_path,
            archive_dir=archive_dir,
            timestamp="20260722-120000",
        )

    assert database_path.read_bytes() == original
    assert archive_path.read_bytes() == b"existing archive"
