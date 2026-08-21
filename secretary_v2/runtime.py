"""Secretary v2 runtime - Pydantic AI Agent definition + tools."""

import asyncio
import inspect
import ipaddress
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ThinkingPart,
)
from pydantic_ai.models import ModelRequestContext

from compaction import (
    _count_tokens,
    _prune_tool_outputs_for_summary,
    maybe_auto_persist_compact,
)
from config import get_config, SECRETARY_PERSONA, DB_SCHEMA_HINT
from fileops import atomic_write_text, edit_snippet, numbered_lines, str_replace_unique
from guardrails import (
    BASE_DIR,
    check_path,
    check_path_decision,
    check_shell_command,
    check_shell_command_decision,
    permission_denied,
    truncate_output,
)
from llm_models import build_model
from market_calendar import get_market_calendar_service
from memory import Database, sanitize_pydantic_messages_for_history

logger = logging.getLogger(__name__)


def _record_agent_event(
    db: Database,
    event_type: str,
    origin: str,
    run_id: Optional[str] = None,
    subject: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort local observability event write.

    Observability must never break the secretary's user-facing behavior.
    """
    try:
        db.create_agent_event(
            event_type=event_type,
            origin=origin,
            run_id=run_id,
            subject=subject,
            payload=payload,
        )
    except Exception as e:
        logger.warning("Failed to record agent event %s: %s", event_type, e)


def _record_permission_denied(deps: "SecretaryDeps", decision) -> str:
    _record_agent_event(
        deps.db,
        "permission_denied",
        origin=deps.origin_channel,
        run_id=deps.run_id or None,
        subject=f"{decision.tool}:{decision.reason}",
        payload={
            "tool": decision.tool,
            "target": decision.target,
            "reason": decision.reason,
            "policy": decision.policy,
            "allowed_alternative": decision.allowed_alternative,
            "message": decision.message,
        },
    )
    return decision.format()


def _local_tz() -> ZoneInfo:
    """Resolve the configured local timezone. Defaults to Asia/Shanghai at config
    load time (see config.py). All LLM-facing time strings flow through this."""
    return ZoneInfo(get_config().timezone)


def _to_local_iso(dt: Optional[datetime], tz: Optional[ZoneInfo] = None) -> str:
    """Render a DB-side datetime (naive UTC) as local-tz ISO with offset suffix.

    DB columns store naive UTC (see memory._utcnow). The LLM sees the result as
    plain text and can't infer the zone, so we always emit an offset-tagged
    string at the injection boundary.
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz or _local_tz()).isoformat()


@dataclass
class SecretaryDeps:
    """Dependencies the agent + tools rely on at run time."""
    db: Database
    agent_id: str = "secretary"
    origin_channel: str = "cli"
    user_id: str = "default"
    conversation_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_key: str = ""
    message_metadata: Optional[Dict[str, Any]] = None
    run_id: str = ""
    skill_content: str = ""
    current_time: str = field(
        default_factory=lambda: datetime.now(_local_tz()).isoformat()
    )
    channels: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[Any] = None  # main.Scheduler — typed Any to avoid circular import
    subagent_registry: Optional[Any] = None  # subagent_runs.SubAgentRegistry


_config = get_config()


def _guard_model_request_context(
    ctx: RunContext[SecretaryDeps], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Keep every request in a tool loop inside the configured context budget."""
    cfg = get_config()
    reserve = cfg.llm.max_tokens + max(2048, int(cfg.history.context_tokens * 0.05))
    input_budget = cfg.history.context_tokens - reserve
    if input_budget <= 0:
        raise RuntimeError(
            "Invalid context budget: history.context_tokens must exceed "
            "llm.max_tokens plus the request safety reserve"
        )

    before_tokens = _count_tokens(request_context.messages)
    if before_tokens <= input_budget:
        return request_context

    max_chars = cfg.history.compact_tool_output_max_chars
    guarded = request_context.messages
    while max_chars > 0 and _count_tokens(guarded) > input_budget:
        guarded = _prune_tool_outputs_for_summary(guarded, max_chars)
        if max_chars <= 125:
            break
        max_chars = max(125, max_chars // 2)

    after_tokens = _count_tokens(guarded)
    if after_tokens > input_budget:
        raise RuntimeError(
            "Current agent run exceeded the safe model input budget "
            f"({after_tokens} > {input_budget} estimated tokens) even after "
            "truncating tool outputs; start a new turn or reduce tool output"
        )
    logger.warning(
        "[run_agent] request context guard truncated tool outputs session=%s tokens=%s->%s budget=%s",
        ctx.deps.session_key,
        before_tokens,
        after_tokens,
        input_budget,
    )
    return replace(request_context, messages=guarded)


_request_guard_hooks = Hooks(before_model_request=_guard_model_request_context)
agent = Agent(
    model=build_model(_config),
    deps_type=SecretaryDeps,
    system_prompt=_config.system_prompt or SECRETARY_PERSONA,
    capabilities=[_request_guard_hooks],
)


def _session_part(value: Optional[str], fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    # Keep keys readable while avoiding separators that make diagnostics awkward.
    return re.sub(r"\s+", "_", text).replace("/", "_")


def build_session_key(
    *,
    agent_id: str = "secretary",
    channel: str = "cli",
    user_id: str = "default",
    conversation_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Build the local history bucket key for one agent/channel conversation."""
    agent = _session_part(agent_id, "secretary")
    origin = _session_part(channel, "cli")
    if origin == "self_test":
        return f"agent:{agent}:self_test:ephemeral:{_session_part(run_id, 'run')}"
    if origin == "scheduled":
        return f"agent:{agent}:scheduled:event:{_session_part(user_id, 'scheduler')}"
    if conversation_id:
        key = f"agent:{agent}:{origin}:conversation:{_session_part(conversation_id)}"
    else:
        key = f"agent:{agent}:{origin}:dm:{_session_part(user_id)}"
    if thread_id:
        key = f"{key}:thread:{_session_part(thread_id)}"
    return key


def history_created_after_for_channel(
    channel: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Return a naive-UTC soft retention cutoff for channel history replay.

    Webhook history uses local calendar days so the replay prefix changes at
    most once per local day. A zero-day setting disables the cutoff. Other
    channels keep their existing token-budget-only behavior.
    """
    cfg = get_config()
    days = cfg.history.webhook_retention_days
    if channel != "http" or days <= 0:
        return None

    tz = ZoneInfo(cfg.timezone)
    local_now = now or datetime.now(tz)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=tz)
    else:
        local_now = local_now.astimezone(tz)
    local_cutoff = local_now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=days - 1)
    return local_cutoff.astimezone(timezone.utc).replace(tzinfo=None)


# Path to the user+agent shared long-term memory scratchpad.
# Injection cap and write soft-cap share one constant so the agent can never
# durably write memory that the prompt injection silently drops.
MEMORY_SOFT_CAP_CHARS = 50_000
# Free-form markdown that gets injected into every run's system prompt.
DEFAULT_MEMORY_FILE = BASE_DIR / "memory.md"
MEMORY_FILE = DEFAULT_MEMORY_FILE
OPEN_EVENTS_CONTEXT_LIMIT = 200

_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_WEEKDAY_EN = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_WEEKDAY_RE = (
    r"(?:周[一二三四五六日天]|星期[一二三四五六日天]|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)"
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(today|tomorrow|tonight|yesterday)\b|今天|今日|今晚|明天|明日|明晚|昨天|昨日|昨晚",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalMention:
    raw: str
    event_date: date
    claimed_weekday: Optional[str] = None


def _local_now_from_deps(deps: SecretaryDeps, tz: ZoneInfo) -> datetime:
    local_now = datetime.fromisoformat(deps.current_time)
    if local_now.tzinfo is None:
        return local_now.replace(tzinfo=tz)
    return local_now.astimezone(tz)


def _weekday_zh(day: date) -> str:
    return _WEEKDAY_ZH[day.weekday()]


def _weekday_en(day: date) -> str:
    return _WEEKDAY_EN[day.weekday()]


def _normalize_claimed_weekday(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    zh_map = {
        "周一": "周一",
        "星期一": "周一",
        "周二": "周二",
        "星期二": "周二",
        "周三": "周三",
        "星期三": "周三",
        "周四": "周四",
        "星期四": "周四",
        "周五": "周五",
        "星期五": "周五",
        "周六": "周六",
        "星期六": "周六",
        "周日": "周日",
        "周天": "周日",
        "星期日": "周日",
        "星期天": "周日",
    }
    if raw in zh_map:
        return zh_map[raw]
    en_map = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return en_map.get(raw.lower(), raw)


def _expected_weekday_for_claim(day: date, claim: str) -> str:
    return _weekday_zh(day) if claim.startswith("周") else _weekday_en(day)


def _coerce_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_temporal_mentions(text: str, *, default_year: int) -> List[TemporalMention]:
    """Extract explicit date mentions and nearby weekday claims from user text.

    This intentionally handles only common reminder shapes. It is a deterministic
    guardrail, not a natural-language date parser.
    """
    if not text:
        return []

    patterns = [
        re.compile(
            rf"(?<!\d)(?P<year>20\d{{2}})[-/](?P<month>0?[1-9]|1[0-2])[-/](?P<day>[12]\d|3[01]|0?[1-9])\s*(?P<weekday>{_WEEKDAY_RE})?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?<![\d/-])(?P<month>0?[1-9]|1[0-2])\s*[/-]\s*(?P<day>[12]\d|3[01]|0?[1-9])\s*(?P<weekday>{_WEEKDAY_RE})?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?<!\d)(?P<month>0?[1-9]|1[0-2])\s*月\s*(?P<day>[12]\d|3[01]|0?[1-9])\s*[日号]?\s*(?P<weekday>{_WEEKDAY_RE})?",
            re.IGNORECASE,
        ),
    ]

    mentions: List[TemporalMention] = []
    seen = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            year = int(match.groupdict().get("year") or default_year)
            month = int(match.group("month"))
            day_num = int(match.group("day"))
            event_date = _coerce_date(year, month, day_num)
            if event_date is None:
                continue
            claim = _normalize_claimed_weekday(match.groupdict().get("weekday"))
            key = (match.start(), match.end(), event_date, claim)
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                TemporalMention(
                    raw=match.group(0).strip(),
                    event_date=event_date,
                    claimed_weekday=claim,
                )
            )
    return mentions


def _temporal_validation_errors(text: str, *, default_year: int) -> List[str]:
    errors = []
    for mention in _extract_temporal_mentions(text, default_year=default_year):
        claim = mention.claimed_weekday
        if not claim:
            continue
        expected = _expected_weekday_for_claim(mention.event_date, claim)
        if claim != expected:
            errors.append(
                f"{mention.raw} claims {claim}, but "
                f"{mention.event_date.isoformat()} is {expected}"
            )
    return errors


def _temporal_note_for_content(
    content: str,
    *,
    default_year: int,
    local_today: date,
) -> str:
    notes = []
    mentions = _extract_temporal_mentions(content or "", default_year=default_year)
    if mentions:
        first = mentions[0]
        if first.event_date > local_today:
            relative = "future"
        elif first.event_date < local_today:
            relative = "past"
        else:
            relative = "today"
        notes.extend(
            [
                f"event_date={first.event_date.isoformat()}",
                f"relative_to_local_today={relative}",
                f"weekday={_weekday_zh(first.event_date)}",
            ]
        )
        if first.claimed_weekday:
            expected = _expected_weekday_for_claim(first.event_date, first.claimed_weekday)
            notes.append(f"claimed_weekday={first.claimed_weekday}")
            if first.claimed_weekday != expected:
                notes.append(f"weekday_mismatch={first.claimed_weekday}->{expected}")
    if _RELATIVE_TIME_RE.search(content or ""):
        notes.append("relative_time_terms=true")
    return " ".join(notes)


def _format_event_context_line(event, *, tz: ZoneInfo, local_today: date) -> str:
    local_created = _to_local_iso(event.created_at, tz)
    if event.created_at is not None:
        created_at = event.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        default_year = created_at.astimezone(tz).year
    else:
        default_year = local_today.year
    temporal_note = _temporal_note_for_content(
        event.content or "",
        default_year=default_year,
        local_today=local_today,
    )
    temporal = f" temporal={temporal_note}" if temporal_note else ""
    source_parts = []
    if getattr(event, "source_channel", None):
        source_parts.append(f"source={event.source_channel}")
    if getattr(event, "session_key", None):
        source_parts.append(f"session={event.session_key}")
    if getattr(event, "source_message_id", None):
        source_parts.append(f"message={event.source_message_id}")
    source = f" {' '.join(source_parts)}" if source_parts else ""
    summary = (getattr(event, "summary", None) or "").strip()
    if not summary:
        summary = (event.content or "")[:160]
    return (
        f"- local_created_at={local_created}: "
        f"[id={event.id} type={event.type} status={event.status}{temporal}{source}] "
        f"{summary}"
    )


def _webhook_record_notice(deps: "SecretaryDeps") -> str:
    metadata = deps.message_metadata or {}
    record = str(metadata.get("record") or "").strip().lower()
    if deps.origin_channel != "http" or record not in {"logged", "open"}:
        return ""
    event_id = metadata.get("recorded_event_id")
    event_ref = f" as event id `{event_id}`" if event_id is not None else ""
    summary_supplied = bool(metadata.get("summary_supplied"))
    update_call = (
        f"`update_event_summary({event_id}, summary)`"
        if event_id is not None
        else "`update_event_summary(event_id, summary)`"
    )
    summary_instruction = (
        "- The webhook supplied a summary; do not update it unless it is clearly misleading.\n"
        if summary_supplied
        else "- If the fallback summary is too raw and you can write a better factual index line, "
        f"call {update_call} once. Summarize the original webhook, not your reply.\n"
    )
    return (
        "### Webhook Record Notice\n"
        f"- This webhook message has already been recorded verbatim in `events`{event_ref} "
        f"with `status='{record}'` before the LLM run.\n"
        f"{summary_instruction}"
        "- Do not call `record_event` to summarize, restate, reclassify, or preserve "
        "the same webhook message for cross-channel visibility.\n"
        "- Only call `record_event` for a distinct actionable follow-up, reminder, "
        "or escalation that is not already fully represented by the recorded message."
    )


def _sql_string_literals(sql: str) -> List[str]:
    literals = []
    for match in re.finditer(r"'((?:''|[^'])*)'|\"((?:\"\"|[^\"])*)\"", sql):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        if value is None:
            continue
        literals.append(value.replace("''", "'").replace('""', '"'))
    return literals


def _language_policy(language: str) -> str:
    normalized = (language or "auto").strip().lower()
    if normalized not in {"zh", "en", "auto"}:
        normalized = "auto"

    if normalized == "en":
        default = "English"
    elif normalized == "auto":
        default = "the user's current language"
    else:
        default = "Chinese"

    return (
        "## Language Policy\n"
        f"- Configured language: `{normalized}`; default user-facing language: {default}.\n"
        "- If the user explicitly asks for a specific language, honor that for the current turn.\n"
        "- Scheduled notifications and proactive messages should use the configured language; "
        "when configured as `auto`, use the language implied by the task prompt and memory.\n"
        "- Do not translate code, file paths, command names, database/table/column names, IDs, "
        "or exact user quotes unless the user asks.\n"
        "- Keep `memory.md` and events concise; preserve user-provided wording when it matters."
    )


def _load_memory_md() -> str:
    """Read memory.md if it exists. Character-cap to keep the prompt sane."""
    memory_path = get_memory_file_path()
    if not memory_path.exists():
        return ""
    try:
        text = memory_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read memory.md: {e}")
        return ""
    # 50,000-character cap: prevents a runaway memory.md from
    # blowing up every prompt. The truncation must be visible to the model,
    # otherwise entries past the cap look like they were never written.
    if len(text) > MEMORY_SOFT_CAP_CHARS:
        return (
            text[:MEMORY_SOFT_CAP_CHARS]
            + "\n\n[memory.md truncated at 50,000 characters — entries beyond "
            "this point are not visible. Prune stale entries with "
            "memory_str_replace.]"
        )
    return text


def _resolve_config_path(path_value: str, *, default_path: Path) -> Path:
    raw = str(path_value or "").strip()
    if not raw or raw == default_path.name:
        return default_path
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path


def get_memory_file_path() -> Path:
    if MEMORY_FILE != DEFAULT_MEMORY_FILE:
        return MEMORY_FILE
    cfg = get_config()
    memory_cfg = getattr(cfg, "memory", None)
    path_value = getattr(memory_cfg, "path", "memory.md")
    return _resolve_config_path(path_value, default_path=MEMORY_FILE)


def _memory_backup_path(local_day: date) -> Path:
    memory_path = get_memory_file_path()
    suffix = memory_path.suffix
    if suffix:
        backup_name = f"{memory_path.stem}-{local_day.isoformat()}{suffix}"
    else:
        backup_name = f"{memory_path.name}-{local_day.isoformat()}"
    return memory_path.with_name(backup_name)


def _local_today_for_config() -> date:
    cfg = get_config()
    try:
        tz = ZoneInfo(cfg.timezone)
    except Exception:
        tz = ZoneInfo("Asia/Shanghai")
    return datetime.now(tz).date()


def ensure_daily_memory_backup() -> Optional[Path]:
    """Create one daily backup next to the configured memory file."""
    cfg = get_config()
    memory_cfg = getattr(cfg, "memory", None)
    if not getattr(memory_cfg, "backup_enabled", True):
        return None
    try:
        memory_path = get_memory_file_path()
        if not memory_path.exists():
            return None
        content = memory_path.read_text(encoding="utf-8")
        if not content.strip():
            return None

        backup_path = _memory_backup_path(_local_today_for_config())
        if backup_path.exists():
            return backup_path

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(content, encoding="utf-8")
        logger.info("Created daily memory backup: %s", backup_path)
        return backup_path
    except Exception as e:
        logger.warning("Failed to create daily memory backup: %s", e)
        return None


def _build_context_layers(deps: SecretaryDeps) -> Tuple[str, str]:
    """Build cache-stable and per-run context as separate layers.

    DeepSeek caches exact input prefixes. Stable policy/schema/skill material must
    therefore precede replayed history, while mutable memory, events, and the
    authoritative clock must follow history as a temporary system message.
    """
    cfg_root = get_config()
    cfg = cfg_root.history
    tz_name = cfg_root.timezone
    tz = ZoneInfo(tz_name)
    local_now = _local_now_from_deps(deps, tz)
    offset = local_now.utcoffset() or timedelta(0)
    offset_sec = int(offset.total_seconds())
    sign = "+" if offset_sec >= 0 else "-"
    abs_h, rem = divmod(abs(offset_sec), 3600)
    abs_m = rem // 60
    offset_display = f"{sign}{abs_h:02d}:{abs_m:02d}"
    if abs_m == 0:
        sql_modifier = f"'{sign}{abs_h} hours'"
    else:
        sql_modifier = f"'{sign}{abs_h} hours', '{sign}{abs_m} minutes'"

    stable_parts = [
        DB_SCHEMA_HINT,
        _language_policy(cfg_root.language),
        "## Trusted runtime context contract\n"
        "- The application supplies a fresh `## Trusted Runtime Context` system block immediately before each new user/task prompt.\n"
        "- Its clock, timezone, local date, and weekday are authoritative over every date claim in conversation history, summaries, memory, events, or user text.\n"
        "- Never calculate a weekday from model memory. Use the supplied weekday or a deterministic tool/database expression.\n"
        "- For trading-day, market-open, or closed-market claims, call `market_calendar`; do not infer from old messages or cron weekdays.\n"
        "- `market_calendar` is factual context only. A non-trading day does not imply `NO_ACTION`;休市 days may still need review reminders, portfolio context, or follow-up.\n"
        "- Historical relative phrases such as today/tomorrow/tonight are quotations, not the current clock.\n"
        "- Before writing reminders/events with a date+weekday, rely on deterministic validation; tools reject mismatched weekday claims.",
    ]
    runtime_parts = []
    open_events_shown = 0
    open_events_total = 0
    recent_events_shown = 0
    recent_events_limit = max(cfg.max_events, 0)

    memory_md = _load_memory_md()
    if memory_md:
        runtime_parts.append(f"## 长期记忆 (memory.md)\n{memory_md}")

    try:
        open_events = deps.db.get_events_by_status("open", limit=OPEN_EVENTS_CONTEXT_LIMIT)
        open_events_total = deps.db.count_events_by_status("open")
        open_event_ids = {event.id for event in open_events}
        event_context_parts = []
        if open_events:
            open_events_shown = len(open_events)
            open_text = "\n".join(
                _format_event_context_line(
                    event,
                    tz=tz,
                    local_today=local_now.date(),
                )
                for event in open_events
            )
            truncated_note = (
                "\n注意：open_events 已截断；完整判断必须 db_query 查询 "
                "`SELECT id,type,status,content,created_at FROM events WHERE status='open' ORDER BY created_at DESC`。"
                if open_events_shown < open_events_total
                else ""
            )
            event_context_parts.append(
                "### 待关注事件 open_events\n"
                f"shown {open_events_shown} / total {open_events_total}。"
                "这些是未闭环、需要持有注意力的对象；不要把 recent_events 当作替代。"
                f"{truncated_note}\n"
                f"{open_text}"
            )

        events = deps.db.get_events_excluding_statuses(
            ["resolved"], limit=cfg.max_events + len(open_event_ids)
        )
        recent_events = [
            event for event in events if event.id not in open_event_ids
        ][:recent_events_limit]
        if recent_events:
            recent_events_shown = len(recent_events)
            events_text = "\n".join(
                _format_event_context_line(
                    event,
                    tz=tz,
                    local_today=local_now.date(),
                )
                for event in recent_events
            )
            event_context_parts.append(
                "### 最近事件片段 recent_events\n"
                f"shown {recent_events_shown} / configured {recent_events_limit}。"
                "这里只是短期连续性片段，已排除上方展示的 open_events；不代表事件全集，"
                "需要完整判断时必须使用 db_query 查询 events 表。\n"
                f"{events_text}"
            )
        if event_context_parts:
            runtime_parts.append("## 事件上下文\n" + "\n\n".join(event_context_parts))
    except Exception as e:
        logger.warning(f"Failed to load recent events: {e}")

    try:
        from skills_loader import get_skills_loader

        loader = get_skills_loader()
        skill_index = loader.get_skill_index()
        if skill_index:
            stable_parts.append(
                "## 可用技能索引\n"
                "下面是可按需加载的项目/用户全局技能索引。索引只说明用途，不等于技能已加载。"
                "当用户请求与某个技能相关，或某个技能明显能提高完成质量时，"
                "先调用 `load_skill(name)` 读取完整技能说明，再继续执行。\n"
                f"{skill_index}"
            )

        auto_skill_parts = []
        max_size = cfg_root.skills.max_size
        for skill_name in loader.get_auto_loaded_skills():
            content = loader.get_skill_content(skill_name)
            if not content:
                continue
            encoded = content.encode("utf-8", errors="replace")
            if len(encoded) > max_size:
                content = encoded[:max_size].decode("utf-8", errors="replace")
                content += f"\n\n... (truncated to {max_size} bytes)"
            auto_skill_parts.append(f"# Skill: {skill_name}\n\n{content}")
        if auto_skill_parts:
            stable_parts.append("## 自动加载技能\n" + "\n\n".join(auto_skill_parts))
    except Exception as e:
        logger.warning(f"Failed to build skill index: {e}")

    registry = getattr(deps, "subagent_registry", None)
    if registry is not None:
        try:
            catalog = registry.agent_catalog()
        except Exception as e:
            logger.warning(f"Failed to build subagent catalog: {e}")
            catalog = []
        if catalog:
            catalog_lines = []
            for agent_def in catalog:
                req = ", ".join(agent_def["required_inputs"]) or "无"
                catalog_lines.append(
                    f"- `{agent_def['name']}` (kind: {agent_def['kind']}) — "
                    f"{agent_def['description']} | 必填输入: {req}"
                )
            stable_parts.append(
                "## 可用后台子任务 (subagents)\n"
                "用 `start_subagent(agent_name, inputs)` 启动后台任务；inputs 必须包含对应的必填输入。"
                "查询/取消/续跑用 `get_subagent_status` / `cancel_subagent` / `resume_subagent`，按 run id 操作。\n"
                + "\n".join(catalog_lines)
            )

    if deps.skill_content:
        runtime_parts.append(f"## 已加载技能\n{deps.skill_content}")

    record_notice = _webhook_record_notice(deps)
    if record_notice:
        runtime_parts.append(record_notice)

    if deps.origin_channel == "http":
        runtime_parts.append(
            "### Webhook Delivery Contract\n"
            "- This run came from an HTTP webhook, not a scheduled task.\n"
            "- Reconcile the webhook payload with relevant facts in `memory.md` before "
            "analyzing it. Unless the current payload explicitly updates a fact, "
            "`memory.md` is authoritative over conflicting or older conversation history.\n"
            "- Do not infer current holdings from tracked/watchlist items or stale history; "
            "only describe an item as a current holding when `memory.md` or the current "
            "payload says it is one.\n"
            "- Analyze the webhook payload and put the user-visible reply in final output; "
            "the application will forward that final output to the configured response channel.\n"
            "- Do not call `send_message` to answer the current webhook, and do not write `NO_ACTION` "
            "as the final output unless the webhook explicitly asks for silent processing."
        )

    runtime_parts.append(
        "## Trusted Runtime Context\n"
        "This block is generated by the application for this run only and is not conversation history.\n"
        f"- now: `{deps.current_time}`\n"
        f"- local_date: `{local_now.date().isoformat()}`\n"
        f"- weekday: `{local_now.strftime('%A')}`\n"
        f"- timezone: `{tz_name}` ({offset_display})\n"
        f"- origin_channel: `{deps.origin_channel}`\n"
        f"- session_key: `{deps.session_key}`\n"
        f"- open_events: shown {open_events_shown} / total {open_events_total}\n"
        f"- recent_events: shown {recent_events_shown} / configured {recent_events_limit}\n"
        "- 完整判断必须 db_query 查询数据库；上方事件上下文是提示片段，不是全集。\n\n"
        "### 时间约定\n"
        f"- 显示时间均为 `{tz_name}` ({offset_display})，带 `{offset_display}` 后缀\n"
        "- DB 的 `created_at` / `updated_at` / `last_run` 列存的是 **UTC**（无后缀）；"
        "`db_query` 返回的字符串也是 UTC\n"
        f"- SQL 中要转本地时间用 `datetime(created_at, {sql_modifier})`\n"
        f"- `schedule_task` 的 cron 表达式按 `{tz_name}` 解释（不是 UTC）"
    )

    return "\n\n".join(stable_parts), "\n\n".join(runtime_parts)


async def dynamic_context(ctx: RunContext[SecretaryDeps]) -> str:
    """Compatibility helper for tests and diagnostics.

    Runtime delivery does not register this as a Pydantic system-prompt runner:
    doing so would place the changing clock before replayed history and destroy
    DeepSeek prefix-cache reuse.
    """
    stable, runtime = _build_context_layers(ctx.deps)
    return f"{stable}\n\n{runtime}"


_SUMMARY_MARKER = "Summary of previous conversation:"


def _cache_optimized_history(
    history: List[ModelMessage],
    stable_context: str,
    runtime_context: str,
) -> List[ModelMessage]:
    """Assemble `stable prefix + history + volatile runtime tail`.

    Old app-managed SystemPromptParts are deliberately removed: legacy history
    may contain a frozen clock. Compaction summaries are retained as stable
    historical context. Both synthetic system layers are passed as history, so
    `result.new_messages()` never persists them back into SQLite.
    """
    summaries: List[SystemPromptPart] = []
    cleaned: List[ModelMessage] = []
    seen_summaries = set()

    for message in history:
        if not isinstance(message, ModelRequest):
            cleaned.append(message)
            continue

        kept_parts = []
        for part in message.parts:
            if not isinstance(part, SystemPromptPart):
                kept_parts.append(part)
                continue
            if _SUMMARY_MARKER in part.content and part.content not in seen_summaries:
                summaries.append(SystemPromptPart(content=part.content))
                seen_summaries.add(part.content)

        if kept_parts:
            cleaned.append(
                ModelRequest(
                    parts=kept_parts,
                    run_id=message.run_id,
                    conversation_id=message.conversation_id,
                    metadata=message.metadata,
                )
            )

    base_prompt = _config.system_prompt or SECRETARY_PERSONA
    prefix = ModelRequest(
        parts=[
            SystemPromptPart(content=base_prompt),
            SystemPromptPart(content=stable_context),
            *summaries,
        ]
    )
    runtime_tail = ModelRequest(parts=[SystemPromptPart(content=runtime_context)])
    return [prefix, *cleaned, runtime_tail]


# ==================== Skill tools ====================


@agent.tool
async def load_skill(ctx: RunContext[SecretaryDeps], name: str) -> str:
    """Load the full content of a discovered local skill by exact name."""
    try:
        from skills_loader import get_skills_loader

        loader = get_skills_loader()
        skills = loader.get_all_skills()
        if name not in skills:
            available = ", ".join(sorted(skills))
            return f"Error: skill not found: {name}. Available skills: {available}"

        content = loader.get_skill_content(name)
        if not content:
            return f"Error: skill has no readable content: {name}"

        max_size = get_config().skills.max_size
        encoded = content.encode("utf-8", errors="replace")
        if len(encoded) > max_size:
            content = encoded[:max_size].decode("utf-8", errors="replace")
            content += f"\n\n... (truncated to {max_size} bytes)"

        parts = [f"# Skill: {name}", "", content]
        resources = loader.get_skill_resources(name)
        if resources:
            # Progressive disclosure: list bundled files so the model can pull
            # only the references it needs via load_skill_file(name, path).
            parts.append("")
            parts.append(
                "Bundled resources (read on demand with "
                f"`load_skill_file(\"{name}\", <path>)`):"
            )
            parts.extend(f"- {rel}" for rel in resources)
        return "\n".join(parts)
    except Exception as e:
        return f"Error loading skill: {e}"


@agent.tool
async def load_skill_file(ctx: RunContext[SecretaryDeps], name: str, path: str) -> str:
    """Read one bundled resource file of a directory skill (e.g. references/x.md).

    Paths are relative to the skill's own directory and confined to it. Use the
    "Bundled resources" list from `load_skill(name)` to see what is available.
    """
    try:
        from skills_loader import get_skills_loader

        loader = get_skills_loader()
        if name not in loader.get_all_skills():
            available = ", ".join(sorted(loader.get_all_skills()))
            return f"Error: skill not found: {name}. Available skills: {available}"

        try:
            content = loader.get_skill_resource(name, path)
        except ValueError as e:
            return f"Error: {e}"
        if content is None:
            resources = loader.get_skill_resources(name)
            listing = ", ".join(resources) if resources else "none"
            return (
                f"Error: resource not found in skill '{name}': {path}. "
                f"Available resources: {listing}"
            )

        max_size = get_config().skills.max_size
        encoded = content.encode("utf-8", errors="replace")
        if len(encoded) > max_size:
            content = encoded[:max_size].decode("utf-8", errors="replace")
            content += f"\n\n... (truncated to {max_size} bytes)"
        return f"# Skill: {name} / {path}\n\n{content}"
    except Exception as e:
        return f"Error loading skill file: {e}"


# ==================== Database tools ====================


@agent.tool
async def db_query(ctx: RunContext[SecretaryDeps], sql: str) -> str:
    """Execute a read-only SQL query (SELECT/PRAGMA only)."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith(("SELECT", "PRAGMA", "WITH")):
        return "Error: Only SELECT, PRAGMA, and WITH (CTE) statements are allowed"
    blocked = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH")
    for kw in blocked:
        if kw in sql_upper.split():
            return f"Error: keyword {kw} is not allowed in db_query"
    try:
        results = ctx.deps.db.execute_query(sql)
        if not results:
            return "Query returned no results"
        return str(results)
    except Exception as e:
        return f"Query error: {e}"


@agent.tool
async def market_calendar(
    ctx: RunContext[SecretaryDeps],
    markets: str = "CN,HK,US",
    date_text: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> str:
    """Check trading days using the configured market calendar service.

    `markets` is a comma-separated list such as "CN,HK,US". For a single-day
    check, pass `date_text` as YYYY-MM-DD. For a range, pass `start` and `end`.
    This tool returns facts only; do not treat non-trading days as automatic
    NO_ACTION because closed-market review reminders can still be useful.
    """
    try:
        cfg = get_config()
        tz = ZoneInfo(cfg.timezone)
        local_today = _local_now_from_deps(ctx.deps, tz).date()
        service = get_market_calendar_service()
        requested_markets = [
            item.strip()
            for item in (markets or ",".join(cfg.market_calendar.markets)).split(",")
            if item.strip()
        ]
        if not requested_markets:
            requested_markets = list(cfg.market_calendar.markets)

        if start or end:
            start_day = date.fromisoformat(start or end or local_today.isoformat())
            end_day = date.fromisoformat(end or start or local_today.isoformat())
            payload = {
                "query": {
                    "start": start_day.isoformat(),
                    "end": end_day.isoformat(),
                    "markets": requested_markets,
                },
                "ranges": [],
                "cache": service.cache_stats(),
            }
            for market in requested_markets:
                window = service.get_range(market, start_day, end_day)
                payload["ranges"].append(
                    {
                        "market": window.market,
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                        "trading_days": [
                            day.isoformat() for day in sorted(window.trading_days)
                        ],
                        "half_trading_days": [
                            day.isoformat()
                            for day in sorted(window.half_trading_days)
                        ],
                        "source": window.source,
                        "degraded": window.degraded,
                        "error": window.error,
                    }
                )
            return json.dumps(payload, ensure_ascii=False, indent=2)

        target = date.fromisoformat(date_text) if date_text else local_today
        payload = {
            "query": {
                "date": target.isoformat(),
                "markets": requested_markets,
            },
            "days": [],
            "cache": service.cache_stats(),
        }
        for market in requested_markets:
            status = service.day_status(market, target)
            payload["days"].append(
                {
                    "market": status.market,
                    "date": status.date.isoformat(),
                    "is_trading_day": status.is_trading_day,
                    "is_half_trading_day": status.is_half_trading_day,
                    "next_trading_day": (
                        status.next_trading_day.isoformat()
                        if status.next_trading_day
                        else None
                    ),
                    "source": status.source,
                    "degraded": status.degraded,
                    "error": status.error,
                }
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        _record_agent_event(
            ctx.deps.db,
            "market_calendar_failed",
            origin=ctx.deps.origin_channel,
            run_id=ctx.deps.run_id or None,
            subject="market_calendar",
            payload={"error": str(e), "markets": markets, "date": date_text},
        )
        return f"Error: market_calendar failed: {e}"


_CONTEXT_VISIBILITY_UPDATE_RE = re.compile(
    r"^\s*UPDATE\s+(?P<table>messages|events)\s+"
    r"SET\s+context_visible\s*=\s*(?:0|1|\?|:[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"WHERE\s+.+?;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _is_context_visibility_only_update(sql: str, table: str) -> bool:
    """Allow a guarded visibility-only UPDATE on an otherwise protected table."""
    statement = sql.strip()
    if ";" in statement.rstrip(";"):
        return False
    match = _CONTEXT_VISIBILITY_UPDATE_RE.fullmatch(statement)
    return bool(match and match.group("table").upper() == table.upper())


@agent.tool
async def db_execute(
    ctx: RunContext[SecretaryDeps], sql: str, params: Optional[List] = None
) -> str:
    """Execute a write SQL statement for ordinary business rows."""
    sql_upper = sql.strip().upper()
    if sql_upper.startswith("SELECT"):
        decision = permission_denied(
            "db_execute",
            "SELECT",
            "wrong_tool",
            policy="db_execute.write_only",
            allowed_alternative="db_query",
            message="Use db_query for SELECT statements",
        )
        return _record_permission_denied(ctx.deps, decision)
    op = sql_upper.split()[0] if sql_upper else ""
    if op in {"DROP", "ATTACH", "DETACH", "ALTER", "CREATE"}:
        decision = permission_denied(
            "db_execute",
            op,
            "blocked_sql_operation",
            policy="db_execute.blocked_operations",
            message=f"{op} operation is blocked",
        )
        return _record_permission_denied(ctx.deps, decision)
    redacted_sql = re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
        "''",
        sql_upper,
    )
    protected_tables = {
        "SCHEDULED_TASKS",
        "MESSAGES",
        "AGENT_EVENTS",
        "SUBAGENT_RUNS",
        "SQLITE_MASTER",
        "SQLITE_SEQUENCE",
    }
    for table in protected_tables:
        if re.search(rf"\b{table}\b", redacted_sql):
            if table == "MESSAGES" and _is_context_visibility_only_update(sql, table):
                continue
            decision = permission_denied(
                "db_execute",
                table.lower(),
                "protected_table",
                policy="db_execute.protected_tables",
                allowed_alternative="dedicated tool or db_query",
                message=f"table {table.lower()} is protected; use the dedicated tool or read-only query",
            )
            return _record_permission_denied(ctx.deps, decision)
    if op in {"INSERT", "UPDATE"} and re.search(r"\bEVENTS\b", redacted_sql):
        tz = _local_tz()
        local_now = _local_now_from_deps(ctx.deps, tz)
        literal_values = _sql_string_literals(sql)
        literal_values.extend(str(value) for value in (params or []) if value is not None)
        text_to_validate = "\n".join(literal_values)
        errors = _temporal_validation_errors(
            text_to_validate,
            default_year=local_now.year,
        )
        if errors:
            _record_agent_event(
                ctx.deps.db,
                "temporal_validation_failed",
                origin=ctx.deps.origin_channel,
                run_id=ctx.deps.run_id or None,
                subject="db_execute:events",
                payload={"errors": errors, "sql_preview": sql[:240]},
            )
            return (
                "Error: temporal validation failed for events write: "
                + "; ".join(errors)
                + ". Correct the absolute date/weekday before writing."
            )
    try:
        affected = ctx.deps.db.execute_statement(sql, params or [])
        return f"Statement executed successfully. Rows affected: {affected}"
    except Exception as e:
        return f"Execution error: {e}"


@agent.tool
async def record_event(
    ctx: RunContext[SecretaryDeps],
    event_type: str,
    content: str,
    status: str = "logged",
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Record a business event with current channel/session provenance.

    Use this instead of raw db_execute INSERTs for reminders, notes, checks, and
    user responses that should remain visible across isolated chat sessions.
    """
    allowed_types = {"remind", "check", "response", "note", "triggered", "subagent"}
    allowed_statuses = {"logged", "open", "resolved", "promoted"}
    event_type = (event_type or "").strip()
    status = (status or "logged").strip()
    if event_type not in allowed_types:
        return f"Error: event_type must be one of {sorted(allowed_types)}"
    if status not in allowed_statuses:
        return f"Error: status must be one of {sorted(allowed_statuses)}"

    tz = _local_tz()
    local_now = _local_now_from_deps(ctx.deps, tz)
    errors = _temporal_validation_errors(content or "", default_year=local_now.year)
    if errors:
        _record_agent_event(
            ctx.deps.db,
            "temporal_validation_failed",
            origin=ctx.deps.origin_channel,
            run_id=ctx.deps.run_id or None,
            subject="record_event",
            payload={"errors": errors, "content_preview": (content or "")[:240]},
        )
        return (
            "Error: temporal validation failed for events write: "
            + "; ".join(errors)
            + ". Correct the absolute date/weekday before writing."
        )

    event_metadata: Dict[str, Any] = {
        "agent_id": ctx.deps.agent_id,
        "conversation_id": ctx.deps.conversation_id,
        "thread_id": ctx.deps.thread_id,
        "sender_id": ctx.deps.user_id,
        "reply_to_id": ctx.deps.reply_to_id,
    }
    if ctx.deps.message_metadata:
        event_metadata["message_metadata"] = ctx.deps.message_metadata
    if metadata:
        event_metadata.update(metadata)

    event = ctx.deps.db.create_event(
        event_type,
        content,
        status=status,
        summary=summary,
        source_channel=ctx.deps.origin_channel,
        session_key=ctx.deps.session_key,
        source_message_id=ctx.deps.reply_to_id,
        metadata=event_metadata,
    )
    _record_agent_event(
        ctx.deps.db,
        "record_event",
        origin=ctx.deps.origin_channel,
        run_id=ctx.deps.run_id or None,
        subject=f"{event_type}:{status}",
        payload={"event_id": event.id, "session_key": ctx.deps.session_key},
    )
    return f"Event recorded: id={event.id} type={event.type} status={event.status}"


@agent.tool
async def update_event_summary(
    ctx: RunContext[SecretaryDeps],
    event_id: int,
    summary: str,
) -> str:
    """Update only events.summary, leaving event content and metadata unchanged.

    For recorded webhooks, use this to replace the automatic fallback summary
    with one factual index line about the original webhook payload.
    """
    if event_id <= 0:
        return "Error: event_id must be positive"
    clean_summary = " ".join(str(summary or "").split())
    if not clean_summary:
        return "Error: summary must be non-empty"

    recorded_id = (ctx.deps.message_metadata or {}).get("recorded_event_id")
    if recorded_id is not None:
        try:
            allowed_id = int(recorded_id)
        except (TypeError, ValueError):
            allowed_id = None
        if allowed_id is not None and event_id != allowed_id:
            return (
                "Error: this webhook run may only update summary for "
                f"recorded_event_id={allowed_id}"
            )

    event = ctx.deps.db.update_event_summary(event_id, clean_summary)
    if event is None:
        return f"Error: event not found: {event_id}"

    _record_agent_event(
        ctx.deps.db,
        "event_summary_update",
        origin=ctx.deps.origin_channel,
        run_id=ctx.deps.run_id or None,
        subject=f"event:{event_id}",
        payload={"summary": event.summary},
    )
    return f"Event summary updated: id={event.id} summary={event.summary}"


# ==================== File tools ====================


def _ensure_memory_document(text: str) -> str:
    if text.strip():
        return text
    return (
        "# Long-Term Memory\n\n"
        "> This is a shared note maintained by the user and the secretary agent.\n\n"
        "## User Preferences\n"
        "<!-- Stable preferences, long-term facts, recurring rules, and focus areas. -->\n\n"
        "## Collaboration Agreements\n"
        "<!-- Notification timing, tone, output style, and review rhythm. -->\n\n"
        "## Tracked Items\n"
        "<!-- Plans, projects, positions, opportunities, or topics that need ongoing tracking. -->\n"
    )


# Serializes read-modify-write cycles across the CLI/Telegram/Feishu/webhook
# channels and scheduled runs that share this process, so concurrent edits
# can't lose each other's updates. Shared by memory_* and file_* mutations.
_file_write_lock = asyncio.Lock()


def _read_memory_document() -> str:
    """Full-fidelity memory.md for editing (no injection cap), with scaffold."""
    memory_path = get_memory_file_path()
    current = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    return _ensure_memory_document(current)


def _memory_capacity_error(updated: str, current: str) -> Optional[str]:
    """Reject writes that grow memory.md past the injection cap.

    Shrinking edits are always allowed even above the cap — pruning must stay
    possible, otherwise an oversized file could never be repaired.
    """
    if len(updated) <= MEMORY_SOFT_CAP_CHARS or len(updated) <= len(current):
        return None
    return (
        f"Error: memory.md would grow to {len(updated)} chars, over the "
        f"{MEMORY_SOFT_CAP_CHARS} char injection cap. Prune stale entries first "
        "(memory_str_replace with new_str='') instead of adding more."
    )


def _write_memory_document(
    ctx: RunContext[SecretaryDeps],
    memory_path: Path,
    updated: str,
    *,
    tool: str,
    subject: str,
) -> None:
    ensure_daily_memory_backup()
    atomic_write_text(memory_path, updated)
    _record_agent_event(
        ctx.deps.db,
        "memory_update",
        origin=ctx.deps.origin_channel,
        run_id=ctx.deps.run_id or None,
        subject=subject,
        payload={"tool": tool, "content_chars": len(updated)},
    )


@agent.tool
async def memory_view(
    ctx: RunContext[SecretaryDeps],
    view_range: Optional[List[int]] = None,
) -> str:
    """View memory.md with line numbers.

    Long-term memory is already injected into every run. Call this before
    editing so memory_str_replace / memory_insert can anchor on the exact
    current on-disk text. view_range=[start, end] limits output to those
    1-based lines (end=-1 means end of file).
    """
    try:
        text = _read_memory_document()
    except Exception as e:
        return f"Error reading memory.md: {e}"
    lines = text.split("\n")

    start, end = 1, len(lines)
    if view_range is not None:
        if len(view_range) != 2:
            return "Error: view_range must be [start_line, end_line]"
        start = max(1, view_range[0])
        end = len(lines) if view_range[1] == -1 else min(view_range[1], len(lines))
        if start > end:
            return f"Error: invalid view_range {view_range}; memory.md has {len(lines)} lines"

    numbered = numbered_lines(lines[start - 1 : end], start)
    warning_chars = get_config().maintenance.memory_warning_chars
    size_summary = (
        f"memory.md size: {len(text)} characters, "
        f"{len(text.encode('utf-8'))} UTF-8 bytes; configured consolidation "
        f"threshold: {warning_chars} characters; hard injection/write cap: "
        f"{MEMORY_SOFT_CAP_CHARS} characters."
    )
    return (
        size_summary
        + "\nContents of memory.md with line numbers:\n"
        + "\n".join(numbered)
    )


@agent.tool
async def memory_str_replace(
    ctx: RunContext[SecretaryDeps],
    old_str: str,
    new_str: str = "",
) -> str:
    """Edit long-term memory by replacing an exact snippet of memory.md.

    This is the primary tool for durable user preferences, collaboration
    agreements, long-term facts, tracking items, and trading plans: rewrite an
    entry by replacing its current text, or delete it with new_str="" (include
    the trailing newline in old_str to avoid leaving a blank line). old_str
    must appear verbatim and exactly once in memory.md — call memory_view
    first and copy the exact text without the line-number prefix. To add a new
    entry, use memory_insert instead.
    """
    if not old_str:
        return "Error: old_str must be a non-empty exact snippet from memory.md"
    try:
        async with _file_write_lock:
            memory_path = get_memory_file_path()
            text = _read_memory_document()

            try:
                updated, changed_line_index = str_replace_unique(
                    text, old_str, new_str, filename="memory.md"
                )
            except ValueError as e:
                return f"Error: {e} Call memory_view and copy the exact current text."

            capacity_error = _memory_capacity_error(updated, text)
            if capacity_error is not None:
                return capacity_error
            _write_memory_document(
                ctx,
                memory_path,
                updated,
                tool="memory_str_replace",
                subject=old_str.strip().split("\n")[0][:80],
            )
        return (
            "memory.md edited. Snippet around the change:\n"
            + edit_snippet(updated, changed_line_index)
        )
    except Exception as e:
        return f"Error updating memory.md: {e}"


@agent.tool
async def memory_insert(
    ctx: RunContext[SecretaryDeps],
    insert_line: int,
    insert_text: str,
) -> str:
    """Add a new entry to memory.md after a given 1-based line number.

    insert_line=N inserts after line N (0 inserts at the top of the file).
    Call memory_view to pick the target line — usually the last entry of the
    section the new item belongs to. Format entries as short "- " bullets and
    keep memory concise. To rewrite or delete existing entries, use
    memory_str_replace instead.
    """
    entry = insert_text.rstrip("\n")
    if not entry.strip():
        return "Error: insert_text must be non-empty"
    try:
        async with _file_write_lock:
            memory_path = get_memory_file_path()
            text = _read_memory_document()
            lines = text.split("\n")

            if insert_line < 0 or insert_line > len(lines):
                return (
                    f"Error: invalid insert_line {insert_line}; "
                    f"it must be within [0, {len(lines)}]"
                )
            # Exact-duplicate guard: the same entry re-inserted (e.g. a retried
            # run) is a silent no-op success rather than a duplicated memory.
            existing = {line.strip() for line in lines if line.strip()}
            if "\n" not in entry and entry.strip() in existing:
                return "memory.md unchanged: identical entry already exists"

            lines[insert_line:insert_line] = entry.split("\n")
            updated = "\n".join(lines)
            if not updated.endswith("\n"):
                updated += "\n"
            capacity_error = _memory_capacity_error(updated, text)
            if capacity_error is not None:
                return capacity_error
            _write_memory_document(
                ctx,
                memory_path,
                updated,
                tool="memory_insert",
                subject=entry.strip().split("\n")[0][:80],
            )
        return (
            "memory.md edited. Snippet around the change:\n"
            + edit_snippet(updated, insert_line)
        )
    except Exception as e:
        return f"Error updating memory.md: {e}"


# file_read truncation cap. Also bounds file_write overwrites: replacing a
# file the model can never have fully read silently drops the unseen tail.
FILE_READ_CAP_CHARS = 50_000


@agent.tool
async def file_read(ctx: RunContext[SecretaryDeps], path: str) -> str:
    """Read a file (whitelist-restricted, max 50KB)."""
    safe, decision = check_path_decision(path, for_write=False, tool="file_read")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)
    file_path = BASE_DIR / safe
    if not file_path.exists():
        return f"Error: File not found: {path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(FILE_READ_CAP_CHARS)
        if len(content) == FILE_READ_CAP_CHARS:
            content += "\n... [truncated at 50KB]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@agent.tool
async def file_edit(
    ctx: RunContext[SecretaryDeps],
    path: str,
    old_str: str,
    new_str: str = "",
) -> str:
    """Edit a file by replacing an exact snippet (whitelist-restricted).

    Prefer this over file_write for targeted changes to an existing data
    file: old_str must appear verbatim and exactly once in the file (call
    file_read first and copy the exact text); new_str="" deletes the snippet.
    For memory.md use memory_str_replace instead.
    """
    safe, decision = check_path_decision(path, for_write=True, tool="file_edit")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)
    if not old_str:
        return f"Error: old_str must be a non-empty exact snippet from {path}"
    file_path = BASE_DIR / safe
    if not file_path.exists():
        return f"Error: File not found: {path}"
    try:
        async with _file_write_lock:
            text = file_path.read_text(encoding="utf-8")
            try:
                updated, changed_line_index = str_replace_unique(
                    text, old_str, new_str, filename=path
                )
            except ValueError as e:
                return f"Error: {e} Call file_read and copy the exact current text."
            atomic_write_text(file_path, updated)
        return (
            f"{path} edited. Snippet around the change:\n"
            + edit_snippet(updated, changed_line_index)
        )
    except Exception as e:
        return f"Error editing file: {e}"


@agent.tool
async def file_write(
    ctx: RunContext[SecretaryDeps],
    path: str,
    content: str,
    mode: str = "overwrite",
) -> str:
    """Write to a file (whitelist-restricted).

    mode='overwrite' replaces the whole file; mode='append' adds to the end.
    Overwriting an existing file larger than the 50KB file_read cap is
    refused — the unseen tail would be silently lost; use file_edit or
    mode='append' instead.
    """
    safe, decision = check_path_decision(path, for_write=True, tool="file_write")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)
    file_path = BASE_DIR / safe
    try:
        async with _file_write_lock:
            existing = (
                file_path.read_text(encoding="utf-8") if file_path.exists() else ""
            )
            if mode == "append":
                updated = existing + content
            else:
                if len(existing) > FILE_READ_CAP_CHARS:
                    return (
                        f"Error: {path} is larger than the 50KB file_read cap; "
                        "overwriting it would silently drop content you never "
                        "saw. Use file_edit for targeted changes or "
                        "mode='append'."
                    )
                updated = content
            atomic_write_text(file_path, updated)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# ==================== HTTP tool ====================


def _is_url_blocked(url: str) -> Optional[str]:
    """Return error message if url targets a private/internal address, else None."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        return f"Invalid URL: {e}"
    host = parsed.hostname
    if not host:
        return "Missing hostname"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"DNS resolution failed: {e}"
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return f"Internal/private address blocked: {ip}"
    return None


@agent.tool
async def http_request(
    ctx: RunContext[SecretaryDeps],
    method: str,
    url: str,
    headers: Optional[Dict] = None,
    body: Optional[str] = None,
) -> str:
    """Send an HTTP request (blocks private IPs via DNS resolution)."""
    import httpx

    block = _is_url_blocked(url)
    if block:
        return f"Error: {block}"

    method_upper = method.upper()
    if method_upper not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"Error: Unsupported method: {method}"

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            resp = await client.request(method_upper, url, headers=headers, content=body)
        return f"Status: {resp.status_code}\nBody: {resp.text[:10_000]}"
    except Exception as e:
        return f"HTTP error: {e}"


# ==================== Search tool ====================


@agent.tool
async def web_search(
    ctx: RunContext[SecretaryDeps], query: str, max_results: int = 5
) -> str:
    """Search the web (Tavily backend; falls back to DDG instant-answer if no key)."""
    cfg = get_config()
    if cfg.search.backend == "tavily" and cfg.search.tavily_api_key:
        return await _search_tavily(query, max_results, cfg.search.tavily_api_key)
    return await _search_duckduckgo(query, max_results)


async def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
    data = resp.json()
    results = [
        f"- [{item.get('title', '')}]({item.get('url', '')}): {item.get('content', '')[:200]}"
        for item in data.get("results", [])
    ]
    answer = data.get("answer", "")
    if answer:
        return f"Answer: {answer}\n\nSources:\n" + "\n".join(results)
    return f"Search results for '{query}':\n" + "\n".join(results) if results else "No results"


async def _search_duckduckgo(query: str, max_results: int) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
        )
    data = resp.json()
    results: List[str] = []
    if data.get("AbstractText"):
        results.append(f"- {data.get('Heading', '')}: {data['AbstractText'][:200]}")
    for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append(f"- {topic['Text'][:200]}")
    if not results:
        return f"No DDG instant answer for '{query}' (configure Tavily API key for real search)"
    return f"Search results for '{query}':\n" + "\n".join(results)


# ==================== Channel tool ====================


@agent.tool
async def send_message(
    ctx: RunContext[SecretaryDeps], text: str, channel: Optional[str] = None
) -> str:
    """Send a message to the user via the origin channel (or an explicit one).

    Keep Telegram/scheduled notification text plain: use short paragraphs and
    simple hyphen bullets, not Markdown headings, bold, tables, or code fences.
    """
    target = channel or ctx.deps.origin_channel
    channel_obj = ctx.deps.channels.get(target)
    if channel_obj is None:
        # For scheduled origin, route to the configured default outgoing channel.
        if target == "scheduled":
            default = get_config().channels.default_outgoing
            channel_obj = ctx.deps.channels.get(default)
            target = default
    if channel_obj is None:
        return f"Error: Channel '{target}' not available"

    # Only forward a routable conversation target. For private chats this may
    # equal the sender id; for group chats it must be the chat/group id.
    forward_user_id = ctx.deps.conversation_id or ctx.deps.user_id
    if ctx.deps.origin_channel == "scheduled":
        forward_user_id = None

    try:
        await channel_obj.send(text, forward_user_id)
        _record_agent_event(
            ctx.deps.db,
            "send_message",
            origin=ctx.deps.origin_channel,
            run_id=ctx.deps.run_id or None,
            subject=target,
            payload={
                "target_channel": target,
                "text_chars": len(text or ""),
                "preview": (text or "")[:160],
                "user_id_forwarded": forward_user_id is not None,
            },
        )
        return f"Message sent via {target}"
    except Exception as e:
        return f"Error sending message: {e}"


# ==================== Schedule tool ====================


@agent.tool
async def schedule_task(
    ctx: RunContext[SecretaryDeps],
    action: str,
    task_id: Optional[str] = None,
    cron: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Manage scheduled tasks. Mutations go through the live scheduler so
    new/changed jobs fire immediately without restart."""
    db = ctx.deps.db
    sched = ctx.deps.scheduler

    if action == "list":
        tasks = [
            task
            for task in db.get_scheduled_tasks(enabled_only=False)
            if task.handler == "agent"
        ]
        if not tasks:
            return "No scheduled tasks found"
        return "\n".join(
            f"- {t.id}: {t.cron} ({'enabled' if t.enabled else 'disabled'}, "
            f"{'protected' if t.protected else 'normal'})"
            for t in tasks
        )

    if action == "create":
        if not task_id or not cron or not prompt:
            return "Error: task_id, cron, and prompt are required for create"
        errors = _temporal_validation_errors(
            prompt,
            default_year=_local_now_from_deps(ctx.deps, _local_tz()).year,
        )
        if errors:
            _record_agent_event(
                ctx.deps.db,
                "temporal_validation_failed",
                origin=ctx.deps.origin_channel,
                run_id=ctx.deps.run_id or None,
                subject="schedule_task:create",
                payload={"task_id": task_id, "errors": errors},
            )
            return (
                "Error: temporal validation failed for scheduled task prompt: "
                + "; ".join(errors)
                + ". Correct the absolute date/weekday before creating the task."
            )
        try:
            db.create_scheduled_task(task_id, cron, prompt)
            if sched is not None:
                sched.add_job(task_id, cron, prompt)
            return f"Task '{task_id}' created"
        except Exception as e:
            return f"Error creating task: {e}"

    if action == "update":
        if not task_id:
            return "Error: task_id is required for update"
        existing = db.get_scheduled_task(task_id)
        if existing is None or existing.handler != "agent" or existing.protected:
            return f"Task '{task_id}' not found or is protected"
        updates: Dict[str, Any] = {}
        if cron:
            updates["cron"] = cron
        if prompt:
            errors = _temporal_validation_errors(
                prompt,
                default_year=_local_now_from_deps(ctx.deps, _local_tz()).year,
            )
            if errors:
                _record_agent_event(
                    ctx.deps.db,
                    "temporal_validation_failed",
                    origin=ctx.deps.origin_channel,
                    run_id=ctx.deps.run_id or None,
                    subject="schedule_task:update",
                    payload={"task_id": task_id, "errors": errors},
                )
                return (
                    "Error: temporal validation failed for scheduled task prompt: "
                    + "; ".join(errors)
                    + ". Correct the absolute date/weekday before updating the task."
                )
            updates["prompt"] = prompt
        if not updates:
            return "Error: at least one of cron or prompt is required for update"
        try:
            task = db.update_scheduled_task(task_id, **updates)
            if not task:
                return f"Task '{task_id}' not found"
            if sched is not None:
                sched.update_job(task_id, cron=cron, prompt=prompt)
            return f"Task '{task_id}' updated"
        except Exception as e:
            return f"Error updating task: {e}"

    if action == "delete":
        if not task_id:
            return "Error: task_id is required for delete"
        existing = db.get_scheduled_task(task_id)
        if existing is None or existing.handler != "agent":
            return f"Task '{task_id}' not found or is protected"
        try:
            ok = db.delete_scheduled_task(task_id)
            if not ok:
                return f"Task '{task_id}' not found or is protected"
            if sched is not None:
                sched.remove_job(task_id)
            return f"Task '{task_id}' deleted"
        except Exception as e:
            return f"Error deleting task: {e}"

    return f"Error: unknown action '{action}'. Valid actions: list, create, update, delete"


# ==================== Subagent tools ====================


@agent.tool
async def start_subagent(
    ctx: RunContext[SecretaryDeps],
    agent_name: str,
    inputs: Dict[str, str],
    engine: Optional[str] = None,
) -> str:
    """Start a non-blocking background subagent run using local codex or claude.

    `agent_name` selects the workflow — see "可用后台子任务 (subagents)" in
    context for available agents and each one's required `inputs` keys (e.g.
    deep_research needs {"topic": "..."}). `engine` may be "codex" or "claude";
    omit it to use the best available CLI. The run executes in the background
    and the user is notified on completion.

    Only call this to START a new run. Never call it for status/progress/cancel/
    resume/list requests about an existing run id — use the other subagent tools.
    """
    registry = ctx.deps.subagent_registry
    if registry is None:
        return "Error: subagent registry is not available"
    payload = {str(k): str(v) for k, v in (inputs or {}).items()}
    payload.setdefault("language", get_config().language)
    subject = (
        payload.get("subject")
        or payload.get("topic")
        or next((v for k, v in payload.items() if k != "language"), "")
    )
    try:
        job_id = registry.start(
            agent_name,
            input_payload=payload,
            subject=subject,
            engine=engine,
            origin_channel=ctx.deps.origin_channel,
            user_id=(
                None
                if ctx.deps.origin_channel == "scheduled"
                else (ctx.deps.conversation_id or ctx.deps.user_id)
            ),
        )
    except (ValueError, RuntimeError) as e:
        return f"Error starting subagent: {e}"
    except Exception as e:
        return f"Error starting subagent: {type(e).__name__}: {e}"
    run = ctx.deps.db.get_subagent_run(job_id)
    engine_text = run.engine if run else (engine or "auto")
    return (
        f"Started subagent `{agent_name}` run `{job_id}` with engine `{engine_text}`. "
        "It will run in the background; I will notify you when it finishes."
    )


@agent.tool
async def get_subagent_status(
    ctx: RunContext[SecretaryDeps], job_id: Optional[str] = None
) -> str:
    """Get one subagent run status, or list recent runs when job_id is omitted.

    Use this for questions like "check sub_xxx status" or "list recent background
    tasks". Works across all subagent kinds. Do not start a new run unless the
    user explicitly asks for one.
    """
    registry = ctx.deps.subagent_registry
    if registry is None:
        return "Error: subagent registry is not available"
    if job_id:
        return registry.status_text(job_id)
    return registry.list_text()


@agent.tool
async def cancel_subagent(ctx: RunContext[SecretaryDeps], job_id: str) -> str:
    """Cancel a running background subagent run by id (any kind)."""
    registry = ctx.deps.subagent_registry
    if registry is None:
        return "Error: subagent registry is not available"
    return (
        f"Subagent run `{job_id}` cancellation requested"
        if registry.cancel(job_id)
        else f"Subagent run `{job_id}` cannot be cancelled or does not exist"
    )


@agent.tool
async def resume_subagent(ctx: RunContext[SecretaryDeps], job_id: str) -> str:
    """Resume a failed/cancelled/pending subagent run from its first incomplete stage (any kind)."""
    registry = ctx.deps.subagent_registry
    if registry is None:
        return "Error: subagent registry is not available"
    return (
        f"Subagent run `{job_id}` resume requested"
        if registry.resume(job_id)
        else f"Subagent run `{job_id}` cannot be resumed or does not exist"
    )


# ==================== Shell tool ====================

# Shell is needed for skills like futuapi/longbridge that drive CLIs and helper
# scripts. Safety: first-token classification + pipe-target check (guardrails),
# subprocess in its own session for clean group kill, hard timeout, output cap.
SHELL_RATE_WINDOW = 60.0
SHELL_RATE_LIMIT = 20
_shell_calls: List[float] = []


def _resolve_shell_cwd(cwd: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if cwd is None or not cwd.strip():
        return str(BASE_DIR), None
    normalized = os.path.normpath(cwd)
    if os.path.isabs(normalized) or normalized.startswith(".."):
        return None, "cwd must stay inside secretary_v2"
    resolved = (BASE_DIR / normalized).resolve()
    try:
        resolved.relative_to(BASE_DIR.resolve())
    except ValueError:
        return None, "cwd must stay inside secretary_v2"
    return str(resolved), None


@agent.tool
async def shell(
    ctx: RunContext[SecretaryDeps],
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> str:
    """Run a shell command. Returns stringified result (exit_code/stdout/stderr or error)."""
    now = time.time()
    _shell_calls[:] = [t for t in _shell_calls if t > now - SHELL_RATE_WINDOW]
    if len(_shell_calls) >= SHELL_RATE_LIMIT:
        decision = permission_denied(
            "shell",
            command,
            "rate_limited",
            policy="shell.rate_limit",
            message="shell rate limit exceeded (20/min)",
        )
        return _record_permission_denied(ctx.deps, decision)
    _shell_calls.append(now)

    decision = check_shell_command_decision(command, tool="shell")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)

    work_dir, cwd_err = _resolve_shell_cwd(cwd)
    if cwd_err:
        decision = permission_denied(
            "shell",
            cwd or "",
            "cwd_escape",
            policy="shell.cwd_within_base_dir",
            message=cwd_err,
        )
        return _record_permission_denied(ctx.deps, decision)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = proc.communicate()
            return (
                f"Error: command timed out after {timeout}s\n"
                f"exit_code: -1\n"
                f"stdout:\n{truncate_output(stdout or '')}\n"
                f"stderr:\n{truncate_output(stderr or '')}"
            )
        return (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{truncate_output(stdout or '')}\n"
            f"stderr:\n{truncate_output(stderr or '')}"
        )
    except Exception as e:
        return f"Error: {e}"


# ==================== run_agent ====================


# LLM call retry policy. Re-throws non-transient errors immediately so we don't
# mask bad-request / auth issues behind retries.
_LLM_MAX_ATTEMPTS = 3
_CONTENT_FILTER_MAX_ATTEMPTS = 2
_LLM_BACKOFF_BASE_SEC = 2.0

# Most recent provider-reported token usage globally and per conversation.
# The global value remains for diagnostics/backward compatibility; /status uses
# the scoped map so concurrent channels cannot show one another's usage.
_last_usage: Optional[Dict[str, Any]] = None
_last_usage_by_session: Dict[str, Dict[str, Any]] = {}


def _usage_details(usage: Any) -> Dict[str, Any]:
    details = getattr(usage, "details", None)
    return details if isinstance(details, dict) else {}


def _last_response_usage(result: Any) -> Optional[Any]:
    """Return the final provider request usage, distinct from run totals."""
    all_messages = getattr(result, "all_messages", None)
    if not callable(all_messages):
        return None
    try:
        messages = all_messages()
    except Exception:
        return None
    for message in reversed(messages or []):
        if isinstance(message, ModelResponse):
            usage = getattr(message, "usage", None)
            if usage is not None:
                return usage
    return None


def _response_usages(messages: Optional[List[ModelMessage]]) -> List[Any]:
    """Return provider usage for each model request in chronological order."""
    if not messages:
        return []
    return [
        message.usage
        for message in messages
        if isinstance(message, ModelResponse) and message.usage is not None
    ]


def _cache_metrics_from_usage(usage: Any) -> Dict[str, Any]:
    """Normalize cache counters from one aggregate or per-request usage object."""
    details = _usage_details(usage)
    input_tokens = getattr(usage, "input_tokens", None)
    cache_read_tokens = getattr(usage, "cache_read_tokens", 0) or 0
    cache_write_tokens = getattr(usage, "cache_write_tokens", 0) or 0
    prompt_cache_hit_tokens = details.get("prompt_cache_hit_tokens")
    prompt_cache_miss_tokens = details.get("prompt_cache_miss_tokens")

    cache_hit_tokens = (
        prompt_cache_hit_tokens
        if isinstance(prompt_cache_hit_tokens, (int, float))
        else cache_read_tokens
    )
    if isinstance(prompt_cache_miss_tokens, (int, float)):
        cache_miss_tokens = prompt_cache_miss_tokens
    elif isinstance(input_tokens, (int, float)):
        cache_miss_tokens = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    else:
        cache_miss_tokens = None

    cache_denom = (
        cache_hit_tokens + cache_miss_tokens
        if isinstance(cache_miss_tokens, (int, float))
        else input_tokens
    )
    cache_hit_ratio = (
        cache_hit_tokens / cache_denom
        if isinstance(cache_denom, (int, float)) and cache_denom > 0
        else None
    )
    return {
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "cache_hit_ratio": cache_hit_ratio,
    }


def _build_usage_payload(
    usage: Any,
    *,
    origin_channel: str,
    at: Optional[str] = None,
    last_request_usage: Optional[Any] = None,
    request_usages: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Normalize Pydantic AI usage plus DeepSeek's OpenAI-compatible extras."""
    details = _usage_details(usage)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cache_metrics = _cache_metrics_from_usage(usage)
    normalized_request_usages = request_usages or []
    if not normalized_request_usages and last_request_usage is not None:
        normalized_request_usages = [last_request_usage]
    request_cache_metrics = [
        {"request": index, **_cache_metrics_from_usage(request_usage)}
        for index, request_usage in enumerate(normalized_request_usages, start=1)
    ]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": getattr(usage, "requests", None),
        **cache_metrics,
        "request_cache_metrics": request_cache_metrics,
        "details": details,
        "last_request_input_tokens": (
            getattr(last_request_usage, "input_tokens", None)
            if last_request_usage is not None
            else None
        ),
        "at": at or datetime.now(_local_tz()).isoformat(timespec="seconds"),
        "origin": origin_channel,
    }


def get_last_usage(session_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the last completed run's cumulative usage and final-request input.

    Run totals aggregate every model request, including tool loops. The
    `last_request_input_tokens` field is the single-request context measurement
    suitable for comparison with the configured context window.
    """
    if session_key is not None:
        return _last_usage_by_session.get(session_key)
    return _last_usage


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True if the error is worth retrying — network, timeout, 5xx, rate limit."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                        httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    # pydantic_ai wraps some errors; inspect message as a soft fallback
    msg = str(exc).lower()
    return any(s in msg for s in ("connection error", "timeout", "temporar", "503", "502", "504"))


def _last_thinking_only_response_text(messages: Optional[List[ModelMessage]]) -> Optional[str]:
    """Return thinking text when the latest model response has no actionable output."""
    if not messages:
        return None
    for msg in reversed(messages):
        if not isinstance(msg, ModelResponse):
            continue
        if msg.parts and all(isinstance(part, ThinkingPart) for part in msg.parts):
            return "\n".join(part.content for part in msg.parts if part.content)
        return None
    return None


def _result_new_messages(result: Any) -> Optional[List[ModelMessage]]:
    try:
        return list(result.new_messages())
    except Exception as e:
        logger.error(f"[run_agent] failed to read new messages: {e}")
        return None


async def run_agent(
    user_text: str,
    db: Database,
    agent_id: str = "secretary",
    origin_channel: str = "cli",
    user_id: str = "default",
    conversation_id: Optional[str] = None,
    reply_to_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    message_metadata: Optional[Dict[str, Any]] = None,
    skill_content: str = "",
    channels: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    subagent_registry: Optional[Any] = None,
) -> str:
    """Serialize a full read/run/write cycle within one conversation."""
    from session_locks import get_session_lock

    lock_key = build_session_key(
        agent_id=agent_id,
        channel=origin_channel,
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        run_id="lock",
    )
    async with get_session_lock(lock_key):
        return await _run_agent_unlocked(
            user_text,
            db,
            agent_id=agent_id,
            origin_channel=origin_channel,
            user_id=user_id,
            conversation_id=conversation_id,
            reply_to_id=reply_to_id,
            thread_id=thread_id,
            message_metadata=message_metadata,
            skill_content=skill_content,
            channels=channels,
            scheduler=scheduler,
            subagent_registry=subagent_registry,
        )


async def _run_agent_unlocked(
    user_text: str,
    db: Database,
    agent_id: str = "secretary",
    origin_channel: str = "cli",
    user_id: str = "default",
    conversation_id: Optional[str] = None,
    reply_to_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    message_metadata: Optional[Dict[str, Any]] = None,
    skill_content: str = "",
    channels: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    subagent_registry: Optional[Any] = None,
) -> str:
    """Run the agent, threading prior conversation in via message_history.

    Persistence flow (this keeps long-running context continuous):
      1. Load recent pydantic-ai messages from DB, bounded by a token budget.
      2. Pass as message_history so the agent sees the running conversation.
      3. After the run, persist result.new_messages() so the next call sees
         the full trajectory (user prompt + tool calls + assistant reply).
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    session_key = build_session_key(
        agent_id=agent_id,
        channel=origin_channel,
        user_id=user_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    deps = SecretaryDeps(
        db=db,
        agent_id=agent_id,
        origin_channel=origin_channel,
        user_id=user_id,
        conversation_id=conversation_id,
        reply_to_id=reply_to_id,
        thread_id=thread_id,
        session_key=session_key,
        message_metadata=message_metadata,
        run_id=run_id,
        skill_content=skill_content,
        current_time=datetime.now(_local_tz()).isoformat(),
        channels=channels or {},
        scheduler=scheduler,
        subagent_registry=subagent_registry,
    )
    _record_agent_event(
        db,
        "run_started",
        origin=origin_channel,
        run_id=run_id,
        subject=user_text[:80],
        payload={
            "user_text_chars": len(user_text or ""),
            "skill_content_loaded": bool(skill_content),
            "agent_id": agent_id,
            "session_key": session_key,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "reply_to_id": reply_to_id,
            "thread_id": thread_id,
        },
    )

    loaded_history_row_ids: List[int] = []
    history_created_after = history_created_after_for_channel(
        origin_channel,
        now=datetime.fromisoformat(deps.current_time),
    )
    persisted_history = db.load_pydantic_messages(
        session_key=session_key,
        include_legacy=False,
        created_after=history_created_after,
        loaded_row_ids=loaded_history_row_ids,
    )
    if (
        not persisted_history
        and origin_channel != "self_test"
        and history_created_after is None
    ):
        loaded_history_row_ids = []
        persisted_history = db.load_pydantic_messages(
            session_key=session_key,
            loaded_row_ids=loaded_history_row_ids,
        )
    pre_run_compact_outcome = None
    try:
        pre_run_compact_outcome = await maybe_auto_persist_compact(
            db,
            history=persisted_history,
            session_key=session_key,
            loaded_row_ids=loaded_history_row_ids,
        )
        if (
            pre_run_compact_outcome
            and pre_run_compact_outcome.changed
            and not pre_run_compact_outcome.failed
        ):
            persisted_history = pre_run_compact_outcome.compacted
    except Exception as e:
        logger.error(f"[run_agent] pre-run compaction failed: {e}")

    replay_history = persisted_history
    stable_context, runtime_context = _build_context_layers(deps)
    history = _cache_optimized_history(
        replay_history,
        stable_context=stable_context,
        runtime_context=runtime_context,
    )
    logger.debug(
        "[run_agent] loaded %s persisted messages; assembled %s messages with cache-stable prefix and runtime tail",
        len(persisted_history),
        len(history),
    )

    async def _call_model(
        prompt: str,
        message_history: List[ModelMessage],
    ) -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
            try:
                return await agent.run(
                    prompt,
                    deps=deps,
                    message_history=message_history,
                )
            except Exception as e:
                last_exc = e
                is_content_filter = isinstance(e, ContentFilterError)
                is_transient = _is_transient_llm_error(e)
                max_attempts = (
                    _CONTENT_FILTER_MAX_ATTEMPTS
                    if is_content_filter
                    else _LLM_MAX_ATTEMPTS
                )
                should_retry = is_content_filter or is_transient
                if attempt >= max_attempts or not should_retry:
                    logger.error(
                        f"Agent run failed (attempt {attempt}/{max_attempts}, "
                        f"transient={is_transient}): {e}"
                    )
                    _record_agent_event(
                        db,
                        "run_failed",
                        origin=origin_channel,
                        run_id=run_id,
                        subject=type(e).__name__,
                        payload={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "transient": is_transient,
                            "error": str(e),
                        },
                    )
                    raise
                if is_content_filter:
                    logger.warning(
                        f"Agent run content filter (attempt {attempt}/{max_attempts}): "
                        f"{type(e).__name__}: {e}; retrying once"
                    )
                    continue
                backoff = _LLM_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"Agent run transient failure (attempt {attempt}/{max_attempts}): "
                    f"{type(e).__name__}: {e}; retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
        # Unreachable due to return/raise above, but keeps type checker happy.
        raise last_exc  # type: ignore[misc]

    result = await _call_model(user_text, history)
    new_msgs = _result_new_messages(result)
    persist_msgs = new_msgs
    response_override: Optional[str] = None

    thinking_only_text = _last_thinking_only_response_text(new_msgs)
    if thinking_only_text:
        if "NO_ACTION" in thinking_only_text.strip().upper():
            response_override = "NO_ACTION"
            persist_msgs = []
        else:
            logger.warning(
                "[run_agent] model returned only thinking; requesting visible text"
            )
            retry_history = [
                *history,
                *sanitize_pydantic_messages_for_history(new_msgs or []),
            ]
            retry_prompt = (
                "Your previous turn produced only reasoning_content/thinking, "
                "which is not a valid final assistant message. Return one visible "
                "assistant message in normal content now. Do not put the answer "
                "only in reasoning/thinking. Do not repeat side-effecting tools "
                "unless strictly necessary."
            )
            result = await _call_model(retry_prompt, retry_history)
            retry_msgs = _result_new_messages(result)
            retry_thinking_only = _last_thinking_only_response_text(retry_msgs)
            if retry_thinking_only:
                if "NO_ACTION" in retry_thinking_only.strip().upper():
                    response_override = "NO_ACTION"
                    persist_msgs = []
                else:
                    error = RuntimeError(
                        "Model returned only reasoning/thinking after visible-output retry"
                    )
                    _record_agent_event(
                        db,
                        "run_failed",
                        origin=origin_channel,
                        run_id=run_id,
                        subject=type(error).__name__,
                        payload={"error": str(error), "thinking_only": True},
                    )
                    raise error
            else:
                persist_msgs = [
                    *sanitize_pydantic_messages_for_history(new_msgs or []),
                    *(retry_msgs or []),
                ]

    usage_payload: Dict[str, Any] = {}
    try:
        # pydantic-ai 1.89 exposes AgentRunResult.usage as a method; newer 1.x
        # makes it a property whose deprecation shim warns when called.
        # Inspect the class so both installs stay warning-free.
        usage_attr = inspect.getattr_static(type(result), "usage", None)
        usage = result.usage if isinstance(usage_attr, property) else result.usage()
        last_request_usage = _last_response_usage(result)
        request_usages = _response_usages(_result_new_messages(result))
        global _last_usage, _last_usage_by_session
        _last_usage = _build_usage_payload(
            usage,
            origin_channel=origin_channel,
            last_request_usage=last_request_usage,
            request_usages=request_usages,
        )
        _last_usage_by_session[session_key] = dict(_last_usage)
        usage_payload = dict(_last_usage)
        logger.info(
            "[run_agent] usage input=%s output=%s total=%s requests=%s cache_read=%s cache_write=%s cache_hit=%s cache_miss=%s cache_requests=%s",
            _last_usage["input_tokens"],
            _last_usage["output_tokens"],
            _last_usage["total_tokens"],
            _last_usage["requests"],
            _last_usage["cache_read_tokens"],
            _last_usage["cache_write_tokens"],
            _last_usage["cache_hit_tokens"],
            _last_usage["cache_miss_tokens"],
            json.dumps(_last_usage["request_cache_metrics"], separators=(",", ":")),
        )
    except Exception as e:
        logger.debug(f"[run_agent] usage unavailable: {e}")

    response = response_override or str(result.output)
    _record_agent_event(
        db,
        "run_finished",
        origin=origin_channel,
        run_id=run_id,
        subject=response[:80],
        payload={
            "response_chars": len(response),
            "no_action": "NO_ACTION" in response.strip().upper(),
            "usage": usage_payload,
        },
    )

    # Startup self-test exercises history preparation plus the full agent.run
    # path, but never persists. Reaching this line means the pipeline works.
    if origin_channel == "self_test":
        logger.info(f"[run_agent] self-test ok (output preview: {response[:60]!r})")
        return response

    # NO_ACTION: don't persist this turn — used by scheduled tasks that found
    # nothing to report, so they don't pollute history.
    if "NO_ACTION" in response.strip().upper():
        _record_agent_event(
            db,
            "scheduled_no_action" if origin_channel == "scheduled" else "no_action",
            origin=origin_channel,
            run_id=run_id,
            subject=user_text[:80],
            payload={"response_preview": response[:160]},
        )
        # Lift to INFO for scheduled origin so log readers can see the task fired
        # and chose to stay silent (vs. failing). Other origins stay debug.
        if origin_channel == "scheduled":
            logger.info(f"[run_agent] scheduled task → NO_ACTION (silent skip)")
        else:
            logger.debug("[run_agent] NO_ACTION — skipping persistence")
        return response

    try:
        if persist_msgs:
            db.save_pydantic_messages(
                persist_msgs,
                agent_id=agent_id,
                session_key=session_key,
                channel=origin_channel,
                conversation_id=conversation_id,
                thread_id=thread_id,
                sender_id=user_id,
                reply_to_id=reply_to_id,
                metadata=message_metadata,
            )
            logger.debug(f"[run_agent] persisted {len(persist_msgs)} new messages")
    except Exception as e:
        logger.error(f"[run_agent] failed to persist messages: {e}")

    if (
        pre_run_compact_outcome
        and pre_run_compact_outcome.changed
        and not pre_run_compact_outcome.failed
    ):
        target = origin_channel
        channel_obj = (channels or {}).get(target)
        if target == "scheduled" or channel_obj is None:
            target = get_config().channels.default_outgoing
            channel_obj = (channels or {}).get(target)
        if channel_obj is not None:
            await channel_obj.send(
                "Conversation history was automatically compacted before this run: "
                f"{pre_run_compact_outcome.before_messages} -> "
                f"{pre_run_compact_outcome.after_messages} messages, "
                f"{pre_run_compact_outcome.before_tokens:,} -> "
                f"{pre_run_compact_outcome.after_tokens:,} tokens.",
                None if origin_channel == "scheduled" else (conversation_id or user_id),
            )

    return response
