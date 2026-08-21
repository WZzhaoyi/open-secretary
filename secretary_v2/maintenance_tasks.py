"""Deterministic P1 built-in maintenance tasks."""

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from builtin_tasks import BuiltinTaskContext, BuiltinTaskResult
from config import get_config
from cron_utils import iter_fire_times


logger = logging.getLogger(__name__)


def _local_now() -> datetime:
    config = get_config()
    return datetime.now(ZoneInfo(config.timezone))


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _memory_path() -> Path:
    from runtime import get_memory_file_path

    return get_memory_file_path()


def _backup_date(memory_path: Path, backup_path: Path) -> Optional[date]:
    pattern = re.compile(
        rf"^{re.escape(memory_path.stem)}-(\d{{4}}-\d{{2}}-\d{{2}})"
        rf"{re.escape(memory_path.suffix)}$"
    )
    match = pattern.fullmatch(backup_path.name)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _memory_health(memory_path: Path, compact_after_chars: int) -> Dict[str, Any]:
    if not memory_path.exists():
        return {
            "path": str(memory_path),
            "exists": False,
            "chars": 0,
            "warning": "memory file is missing",
        }
    text = memory_path.read_text(encoding="utf-8")
    chars = len(text)
    return {
        "path": str(memory_path),
        "exists": True,
        "chars": chars,
        "warning": (
            f"memory size {chars} exceeds recommended limit {compact_after_chars}"
            if chars > compact_after_chars
            else None
        ),
    }


def _backup_health(
    memory_path: Path,
    *,
    local_today: date,
    enabled: bool,
    max_age_days: int,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "latest": None,
            "age_days": None,
            "warning": None,
        }
    if (
        not memory_path.exists()
        or not memory_path.read_text(encoding="utf-8").strip()
    ):
        return {"enabled": True, "latest": None, "age_days": None, "warning": None}

    latest_day = None
    latest_path = None
    pattern = f"{memory_path.stem}-*{memory_path.suffix}"
    for candidate in memory_path.parent.glob(pattern):
        backup_day = _backup_date(memory_path, candidate)
        if backup_day is not None and (latest_day is None or backup_day > latest_day):
            latest_day = backup_day
            latest_path = candidate
    if latest_day is None:
        return {
            "enabled": True,
            "latest": None,
            "age_days": None,
            "warning": "memory backup is missing",
        }

    age_days = (local_today - latest_day).days
    return {
        "enabled": True,
        "latest": str(latest_path),
        "age_days": age_days,
        "warning": (
            f"latest memory backup is {age_days} days old"
            if age_days > max_age_days
            else None
        ),
    }


async def context_maintenance(ctx: BuiltinTaskContext) -> BuiltinTaskResult:
    """Hide expired messages and closed events without deleting history."""
    config = get_config()
    policy = config.maintenance
    now_local = _local_now()
    message_cutoff = _naive_utc(
        now_local - timedelta(days=policy.message_hide_after_days)
    )
    event_cutoff = _naive_utc(
        now_local - timedelta(days=policy.event_archive_after_days)
    )

    hidden_messages = ctx.db.archive_messages_before(message_cutoff)
    archived_events = ctx.db.archive_closed_events_before(event_cutoff)

    backlog = ctx.db.get_maintenance_backlog(
        message_cutoff,
        event_before=event_cutoff,
    )

    details = {
        "message_cutoff": message_cutoff.isoformat(),
        "event_cutoff": event_cutoff.isoformat(),
        "hidden_messages": hidden_messages,
        "archived_events": archived_events,
        "backlog": backlog,
    }
    ctx.db.create_agent_event(
        "context_maintenance_finished",
        origin="scheduled",
        subject=ctx.task_id,
        payload=details,
    )
    changed = hidden_messages > 0 or archived_events > 0
    return BuiltinTaskResult(
        status="succeeded" if changed else "skipped",
        details=details,
    )


def _scheduled_success_counts(
    ctx: BuiltinTaskContext,
    since_utc: datetime,
) -> Dict[str, int]:
    rows = ctx.db.execute_query(
        "SELECT subject, COUNT(*) AS successes "
        "FROM agent_events "
        "WHERE type='scheduled_task_succeeded' AND created_at >= ? "
        "GROUP BY subject",
        [since_utc],
    )
    return {
        str(row["subject"]): int(row["successes"])
        for row in rows
        if row.get("subject")
    }


def _observation_start(ctx: BuiltinTaskContext, fallback: datetime) -> datetime:
    rows = ctx.db.execute_query(
        "SELECT MIN(created_at) AS first_attempt FROM agent_events "
        "WHERE type='scheduled_task_started'"
    )
    if not rows or not rows[0].get("first_attempt"):
        return fallback
    value = datetime.fromisoformat(str(rows[0]["first_attempt"]))
    return max(fallback, value)


async def system_health_review(ctx: BuiltinTaskContext) -> BuiltinTaskResult:
    """Compare expected cron fires with structured outcomes and inspect backlog."""
    config = get_config()
    policy = config.maintenance
    now_local = _local_now()
    fallback_start_utc = _naive_utc(
        now_local - timedelta(days=policy.health_lookback_days)
    )
    observation_start_utc = _observation_start(ctx, fallback_start_utc)
    observation_start_local = observation_start_utc.replace(
        tzinfo=timezone.utc
    ).astimezone(ZoneInfo(config.timezone))
    success_counts = _scheduled_success_counts(ctx, observation_start_utc)

    missed = []
    failed = []
    for task in ctx.db.get_scheduled_tasks(enabled_only=True):
        if task.id == ctx.task_id:
            continue
        task_start_local = observation_start_local
        if task.created_at is not None:
            created_local = task.created_at.replace(
                tzinfo=timezone.utc
            ).astimezone(ZoneInfo(config.timezone))
            task_start_local = max(task_start_local, created_local)
        expected = sum(
            1
            for _ in iter_fire_times(
                task.cron,
                config.timezone,
                task_start_local,
                now_local,
            )
        )
        observed = success_counts.get(task.id, 0)
        if observed < expected:
            missed.append(
                {
                    "task_id": task.id,
                    "expected": expected,
                    "succeeded": observed,
                }
            )
        if task.last_error:
            failed.append({"task_id": task.id, "error": task.last_error})

    context_cutoff = _naive_utc(
        now_local - timedelta(days=policy.message_hide_after_days)
    )
    event_cutoff = _naive_utc(
        now_local - timedelta(days=policy.event_archive_after_days)
    )
    backlog = ctx.db.get_maintenance_backlog(
        context_cutoff,
        event_before=event_cutoff,
    )
    memory_path = _memory_path()
    memory = _memory_health(memory_path, policy.memory_warning_chars)
    backup = _backup_health(
        memory_path,
        local_today=now_local.date(),
        enabled=config.memory.backup_enabled,
        max_age_days=policy.health_lookback_days,
    )
    subagent_failures = ctx.db.execute_query(
        "SELECT created_at, run_id, subject FROM agent_events "
        "WHERE type='subagent_step_finished' "
        "AND payload_json LIKE '%\"status\": \"failed\"%' "
        "AND created_at >= ? ORDER BY created_at DESC LIMIT 5",
        [fallback_start_utc],
    )

    anomalies = []
    if missed:
        anomalies.append(
            "Scheduled: "
            + "; ".join(
                f"{item['task_id']} {item['succeeded']}/{item['expected']}"
                for item in missed[:8]
            )
        )
    if failed:
        anomalies.append(
            "Failures: "
            + "; ".join(
                f"{item['task_id']}: {item['error'][:120]}"
                for item in failed[:5]
            )
        )
    if backlog["old_messages"] or backlog["old_archivable_events"]:
        anomalies.append(
            "Context backlog: "
            f"{backlog['old_messages']} old active messages, "
            f"{backlog['old_archivable_events']} closed events"
        )
    if memory["warning"]:
        anomalies.append(f"Memory: {memory['warning']}")
    if backup["warning"]:
        anomalies.append(f"Backup: {backup['warning']}")
    if subagent_failures:
        anomalies.append(f"Subagent: {len(subagent_failures)} recent failed stage(s)")

    details = {
        "observation_start": observation_start_utc.isoformat(),
        "missed": missed,
        "failed": failed,
        "backlog": backlog,
        "memory": memory,
        "backup": backup,
        "subagent_failures": subagent_failures,
        "alerted": bool(anomalies),
    }
    if anomalies:
        if ctx.notify is None:
            raise RuntimeError(
                "system health anomalies found but notifier is unavailable"
            )
        await ctx.notify("Secretary system health review\n- " + "\n- ".join(anomalies))

    ctx.db.create_agent_event(
        "system_health_review_finished",
        origin="scheduled",
        subject=ctx.task_id,
        payload=details,
    )
    return BuiltinTaskResult(
        status="succeeded" if anomalies else "skipped",
        details=details,
    )
