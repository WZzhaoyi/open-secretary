"""Shared cron parsing and occurrence calculation."""

import re
from datetime import datetime
from typing import Iterator

from apscheduler.triggers.cron import CronTrigger


_UNIX_DOW_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def translate_unix_dow(field: str) -> str:
    """Translate Unix numeric weekdays to APScheduler weekday names."""
    return re.sub(
        r"(?<!/)\b[0-7]\b",
        lambda match: _UNIX_DOW_NAMES[int(match.group(0))],
        field,
    )


def build_cron_trigger(cron: str, timezone: str) -> CronTrigger:
    """Build an APScheduler trigger from the project's five-field cron syntax."""
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"cron must have 5 fields, got {len(parts)}: {cron!r}")
    return CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=translate_unix_dow(parts[4]),
        timezone=timezone,
    )


def iter_fire_times(
    cron: str,
    timezone: str,
    start: datetime,
    end: datetime,
) -> Iterator[datetime]:
    """Yield cron fire times in the inclusive aware-datetime window."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("cron occurrence bounds must be timezone-aware")
    if end < start:
        return

    trigger = build_cron_trigger(cron, timezone)
    previous = None
    cursor = start
    while True:
        next_fire = trigger.get_next_fire_time(previous, cursor)
        if next_fire is None or next_fire > end:
            return
        yield next_fire
        previous = next_fire
        cursor = next_fire
