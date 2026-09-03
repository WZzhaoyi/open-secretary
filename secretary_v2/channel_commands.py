"""Channel-independent implementations of built-in chat commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from i18n import t

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandScope:
    """Identity needed to address one isolated conversation history bucket."""

    channel: str
    user_id: str
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    agent_id: str = "secretary"

    def session_key(self) -> str:
        from runtime import build_session_key

        return build_session_key(
            agent_id=self.agent_id,
            channel=self.channel,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            thread_id=self.thread_id,
        )


def _replayable_history(db, scope: CommandScope):
    """Mirror run_agent's session/legacy history selection exactly."""
    from runtime import history_created_after_for_channel

    session_key = scope.session_key()
    created_after = history_created_after_for_channel(scope.channel)
    history = db.load_pydantic_messages(
        session_key=session_key,
        include_legacy=False,
        created_after=created_after,
    )
    if not history and created_after is None:
        history = db.load_pydantic_messages(session_key=session_key)
    return history


def _cache_counts(metrics) -> tuple[int, int]:
    hit = metrics.get("cache_hit_tokens", 0)
    miss = metrics.get("cache_miss_tokens", 0)
    return (
        int(hit) if isinstance(hit, (int, float)) else 0,
        int(miss) if isinstance(miss, (int, float)) else 0,
    )


def _usage_window_totals(events, *, since: datetime):
    runs = input_tokens = output_tokens = total_tokens = 0
    cache_runs = hit = miss = 0
    for event in events:
        if event.created_at is None or event.created_at < since:
            continue
        try:
            payload = json.loads(event.payload_json or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            continue

        sample_input = usage.get("input_tokens")
        sample_output = usage.get("output_tokens")
        sample_total = usage.get("total_tokens")
        has_tokens = any(
            isinstance(value, (int, float))
            for value in (sample_input, sample_output, sample_total)
        )
        if has_tokens:
            normalized_input = (
                int(sample_input) if isinstance(sample_input, (int, float)) else 0
            )
            normalized_output = (
                int(sample_output) if isinstance(sample_output, (int, float)) else 0
            )
            normalized_total = (
                int(sample_total)
                if isinstance(sample_total, (int, float))
                else normalized_input + normalized_output
            )
            runs += 1
            input_tokens += normalized_input
            output_tokens += normalized_output
            total_tokens += normalized_total

        sample_hit, sample_miss = _cache_counts(usage)
        if sample_hit + sample_miss > 0:
            cache_runs += 1
            hit += sample_hit
            miss += sample_miss
    return runs, input_tokens, output_tokens, total_tokens, cache_runs, hit, miss


def _format_cache_window(sample, lang: str) -> str:
    _, _, _, _, runs, hit, miss = sample
    if runs <= 0 or hit + miss <= 0:
        return t("command.status.no_cache_window_metrics", lang)
    return t(
        "command.status.cache_window_metrics",
        lang,
        runs=runs,
        ratio=f"{hit / (hit + miss) * 100:.1f}%",
    )


def _format_token_window(sample, lang: str) -> str:
    runs, input_tokens, output_tokens, total_tokens, _, _, _ = sample
    if runs <= 0:
        return t("command.status.no_token_window_metrics", lang)
    return t(
        "command.status.token_window_metrics",
        lang,
        runs=runs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _format_last_run_cache(usage, lang: str) -> str:
    if not usage:
        return t("command.status.no_run", lang)
    hit, miss = _cache_counts(usage)
    write = usage.get("cache_write_tokens") or 0
    if hit + miss <= 0 and not write:
        return t("command.status.no_cache_metrics", lang)
    ratio = f"{hit / (hit + miss) * 100:.1f}%" if hit + miss > 0 else "?"
    return t(
        "command.status.cache_last_run",
        lang,
        cache_hit=hit,
        cache_miss=miss,
        cache_write=int(write),
        ratio=ratio,
    )


def build_status_text(
    *,
    scope: CommandScope,
    lang: str,
    peer_channel_names: Iterable[str],
    channel_health: Optional[str] = None,
) -> str:
    """Build /status from real config, current-session history, and last usage."""
    from config import get_config
    from memory import get_db
    from runtime import get_last_usage, get_memory_file_path

    cfg = get_config()
    db = get_db()
    session_key = scope.session_key()
    try:
        stats = db.get_message_stats()
        history_count = len(_replayable_history(db, scope))
        memory_path: Path = get_memory_file_path()
        memory_status = (
            f"{memory_path.stat().st_size} bytes"
            if memory_path.exists()
            else t("command.status.memory_missing", lang)
        )
    except Exception as e:
        logger.error("Failed to build status database/memory metrics: %s", e)
        history_count = -1
        memory_status = t("command.status.read_failed", lang)
        stats = {"total_messages": -1}

    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        since_24h = now - timedelta(hours=24)
        events = db.get_agent_events_since(
            since_24h,
            event_type="run_finished",
        )
        usage_24h = _usage_window_totals(events, since=since_24h)
        token_24h_status = _format_token_window(usage_24h, lang)
        cache_24h_status = _format_cache_window(usage_24h, lang)
    except Exception as e:
        logger.error("Failed to build 24h usage metrics: %s", e)
        token_24h_status = t("command.status.read_failed", lang)
        cache_24h_status = t("command.status.read_failed", lang)

    usage = get_last_usage(session_key=session_key)
    last_cache_status = _format_last_run_cache(usage, lang)
    window = cfg.history.context_tokens
    if usage:
        request_input = usage.get("last_request_input_tokens")
        # A one-request RunUsage is also an exact single-request measurement.
        if request_input is None and usage.get("requests") == 1:
            request_input = usage.get("input_tokens")
        if request_input is not None:
            pct = f"{request_input / window * 100:.0f}%" if window else "?"
            usage_status = t(
                "command.status.usage",
                lang,
                input_tokens=request_input,
                window=window,
                pct=pct,
                at=usage.get("at", "?"),
                origin=usage.get("origin", "?"),
            )
        elif usage.get("input_tokens") is not None:
            usage_status = t(
                "command.status.run_usage",
                lang,
                input_tokens=usage["input_tokens"],
                requests=usage.get("requests", "?"),
                at=usage.get("at", "?"),
                origin=usage.get("origin", "?"),
            )
        else:
            usage_status = t("command.status.no_run", lang)

    else:
        usage_status = t("command.status.no_run", lang)

    hist_cfg = cfg.history
    compact_threshold = int(hist_cfg.context_tokens * hist_cfg.compress_threshold)
    auto_compact_status = t(
        "command.status.enabled" if hist_cfg.auto_compact else "command.status.disabled",
        lang,
    )
    channels_str = ", ".join(f"`{name}`" for name in peer_channel_names)
    return t(
        "command.status",
        lang,
        model=f"{cfg.llm.provider}/{cfg.llm.model}",
        timezone=cfg.timezone,
        channels=channels_str,
        channel_health=channel_health or t("command.status.health_unknown", lang),
        total_messages=stats.get("total_messages", "?"),
        history_count=history_count,
        usage_status=usage_status,
        token_24h_status=token_24h_status,
        last_cache_status=last_cache_status,
        cache_24h_status=cache_24h_status,
        compact_threshold=compact_threshold,
        tail_budget=hist_cfg.tail_token_budget,
        auto_compact_status=auto_compact_status,
        min_messages=hist_cfg.compact_min_active_messages,
        cooldown_minutes=hist_cfg.compact_cooldown_minutes,
        tool_output_chars=hist_cfg.compact_tool_output_max_chars,
        memory_status=memory_status,
    )


async def compact_conversation(*, scope: CommandScope, lang: str) -> str:
    """Compact the scoped conversation and render a localized result."""
    from compaction import force_compact
    from memory import get_db
    from runtime import history_created_after_for_channel
    from session_locks import get_session_lock

    session_key = scope.session_key()
    created_after = history_created_after_for_channel(scope.channel)
    compact_kwargs = {"session_key": session_key}
    if created_after is not None:
        compact_kwargs.update(
            created_after=created_after,
            include_legacy=False,
        )
    async with get_session_lock(session_key):
        result = await force_compact(get_db(), **compact_kwargs)
    if result.status == "completed":
        return t(
            "command.compact.completed",
            lang,
            before=result.before_messages,
            after=result.after_messages,
        )
    if result.status == "not_needed":
        return t("command.compact.not_needed", lang)
    if result.status == "not_enough_history":
        return t("command.compact.not_enough_history", lang)
    if result.status == "partial_failure":
        return t("command.compact.partial_failure", lang, error=result.error or "?")
    return t("command.compact.failed", lang, error=result.error or "unknown error")
