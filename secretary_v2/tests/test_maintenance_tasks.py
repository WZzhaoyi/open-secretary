"""Tests for deterministic P1 maintenance tasks."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from builtin_tasks import BuiltinTaskContext
from cron_utils import iter_fire_times


def _backdate_event(db, event_id: int, created_at: datetime) -> None:
    db.execute_statement(
        "UPDATE events SET created_at=? WHERE id=?",
        [created_at, event_id],
    )


def _write_memory(path, text: str = "durable memory") -> None:
    path.write_text(text, encoding="utf-8")


def test_iter_fire_times_uses_unix_sunday_and_inclusive_window():
    timezone = ZoneInfo("Asia/Shanghai")
    start = datetime(2026, 8, 16, 0, 0, tzinfo=timezone)
    end = datetime(2026, 8, 16, 23, 59, tzinfo=timezone)

    fires = list(iter_fire_times("30 4 * * 0", "Asia/Shanghai", start, end))

    assert fires == [datetime(2026, 8, 16, 4, 30, tzinfo=timezone)]


def test_archive_closed_events_preserves_logged_open_and_recent(test_db):
    cutoff = datetime(2026, 8, 14)
    old = cutoff - timedelta(days=1)
    resolved = test_db.create_event("note", "resolved", status="resolved")
    promoted = test_db.create_event("note", "promoted", status="promoted")
    logged = test_db.create_event("note", "logged", status="logged")
    open_event = test_db.create_event("note", "open", status="open")
    recent = test_db.create_event("note", "recent resolved", status="resolved")
    for event in (resolved, promoted, logged, open_event):
        _backdate_event(test_db, event.id, old)

    affected = test_db.archive_closed_events_before(cutoff)

    assert affected == 2
    rows = {
        row["id"]: row
        for row in test_db.execute_query(
            "SELECT id,status,context_visible FROM events ORDER BY id"
        )
    }
    assert rows[resolved.id]["context_visible"] == 0
    assert rows[promoted.id]["context_visible"] == 0
    assert rows[logged.id]["context_visible"] == 1
    assert rows[open_event.id]["context_visible"] == 1
    assert rows[recent.id]["context_visible"] == 1


@pytest.mark.asyncio
async def test_context_maintenance_hides_expired_messages_without_summarizing(
    test_db, monkeypatch
):
    import compaction
    import maintenance_tasks

    now = datetime(2026, 8, 21, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    old = datetime(2026, 8, 1)
    session_key = "agent:secretary:http:conversation:user"
    for index in range(3):
        test_db.save_pydantic_messages(
            [ModelRequest(parts=[UserPromptPart(content=f"message {index}")])],
            session_key=session_key,
            channel="http",
        )
    message_rows = test_db.execute_query(
        "SELECT id FROM messages WHERE session_key=? ORDER BY id",
        [session_key],
    )
    for row in message_rows[:2]:
        test_db.execute_statement(
            "UPDATE messages SET created_at=? WHERE id=?",
            [old, row["id"]],
        )

    resolved = test_db.create_event("note", "done", status="resolved")
    promoted = test_db.create_event("note", "remembered", status="promoted")
    logged = test_db.create_event("note", "log", status="logged")
    open_event = test_db.create_event("note", "pending", status="open")
    for event in (resolved, promoted, logged, open_event):
        _backdate_event(test_db, event.id, old)

    async def forbidden_chat_summary(*_args, **_kwargs):
        raise AssertionError("chat summarizer must not be called")

    def forbidden_memory_access():
        raise AssertionError("context maintenance must not access memory.md")

    monkeypatch.setattr(maintenance_tasks, "_local_now", lambda: now)
    monkeypatch.setattr(maintenance_tasks, "_memory_path", forbidden_memory_access)
    monkeypatch.setattr(compaction, "force_compact", forbidden_chat_summary)

    result = await maintenance_tasks.context_maintenance(
        BuiltinTaskContext(task_id="context_maintenance", db=test_db)
    )

    assert result.status == "succeeded"
    assert result.details.keys() >= {
        "hidden_messages",
        "archived_events",
        "backlog",
    }
    assert "memory" not in result.details
    assert "backup" not in result.details
    message_visibility = [
        row["context_visible"]
        for row in test_db.execute_query(
            "SELECT context_visible FROM messages "
            "WHERE session_key=? ORDER BY id",
            [session_key],
        )
    ]
    assert message_visibility == [0, 0, 1]
    rows = {
        row["id"]: row
        for row in test_db.execute_query(
            "SELECT id,status,context_visible FROM events ORDER BY id"
        )
    }
    assert rows[resolved.id]["context_visible"] == 0
    assert rows[promoted.id]["context_visible"] == 0
    assert rows[logged.id]["context_visible"] == 1
    assert rows[open_event.id]["context_visible"] == 1
    assert test_db.get_agent_events()[0].type == "context_maintenance_finished"


@pytest.mark.asyncio
async def test_system_health_review_alerts_for_missed_cron_without_agent(
    test_db, tmp_path, monkeypatch
):
    import maintenance_tasks
    from config import get_config

    now = datetime(2026, 8, 21, 18, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    task = test_db.create_scheduled_task(
        "daily_check",
        "0 8 * * *",
        "agent prompt",
    )
    test_db.execute_statement(
        "UPDATE scheduled_tasks SET created_at=? WHERE id=?",
        [datetime(2026, 8, 14), task.id],
    )
    memory_path = tmp_path / "memory.md"
    _write_memory(memory_path)
    alerts = []

    async def notify(text):
        alerts.append(text)

    config = get_config()
    monkeypatch.setattr(config.memory, "backup_enabled", False)
    monkeypatch.setattr(maintenance_tasks, "_local_now", lambda: now)
    monkeypatch.setattr(maintenance_tasks, "_memory_path", lambda: memory_path)

    result = await maintenance_tasks.system_health_review(
        BuiltinTaskContext(
            task_id="system_review",
            db=test_db,
            notify=notify,
        )
    )

    assert result.status == "succeeded"
    assert result.details["missed"][0]["task_id"] == "daily_check"
    assert result.details["missed"][0]["expected"] > 0
    assert alerts and "daily_check" in alerts[0]
    assert test_db.get_agent_events()[0].type == "system_health_review_finished"


@pytest.mark.asyncio
async def test_system_health_review_is_silent_when_no_anomaly(
    test_db, tmp_path, monkeypatch
):
    import maintenance_tasks
    from config import get_config

    now = datetime(2026, 8, 21, 18, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    task = test_db.create_scheduled_task(
        "completed_check",
        "0 17 * * *",
        "agent prompt",
    )
    test_db.execute_statement(
        "UPDATE scheduled_tasks SET created_at=? WHERE id=?",
        [datetime(2026, 8, 21, 8, 0), task.id],
    )
    success = test_db.create_agent_event(
        "scheduled_task_succeeded",
        origin="scheduled",
        subject=task.id,
    )
    test_db.execute_statement(
        "UPDATE agent_events SET created_at=? WHERE id=?",
        [datetime(2026, 8, 21, 9, 0), success.id],
    )
    memory_path = tmp_path / "memory.md"
    _write_memory(memory_path)
    alerts = []

    async def notify(text):
        alerts.append(text)

    config = get_config()
    monkeypatch.setattr(config.memory, "backup_enabled", False)
    monkeypatch.setattr(maintenance_tasks, "_local_now", lambda: now)
    monkeypatch.setattr(maintenance_tasks, "_memory_path", lambda: memory_path)

    result = await maintenance_tasks.system_health_review(
        BuiltinTaskContext(
            task_id="system_review",
            db=test_db,
            notify=notify,
        )
    )

    assert result.status == "skipped"
    assert not result.details["alerted"]
    assert alerts == []
