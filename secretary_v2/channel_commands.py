"""Channel-independent implementations of built-in chat commands."""

from __future__ import annotations

from dataclasses import dataclass
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


def _replayable_history(db, session_key: str):
    """Mirror run_agent's session/legacy history selection exactly."""
    history = db.load_pydantic_messages(
        session_key=session_key,
        include_legacy=False,
    )
    if not history:
        history = db.load_pydantic_messages(session_key=session_key)
    return history


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
        history_count = len(_replayable_history(db, session_key))
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

    usage = get_last_usage(session_key=session_key)
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

        cache_hit = usage.get("cache_hit_tokens")
        cache_miss = usage.get("cache_miss_tokens")
        cache_read = usage.get("cache_read_tokens") or 0
        cache_write = usage.get("cache_write_tokens") or 0
        cache_ratio = usage.get("cache_hit_ratio")
        if cache_hit or cache_miss or cache_read or cache_write:
            ratio = (
                f"{cache_ratio * 100:.1f}%"
                if isinstance(cache_ratio, (int, float))
                else "?"
            )
            cache_status = t(
                "command.status.cache_metrics",
                lang,
                cache_hit=int(cache_hit or 0),
                cache_miss=int(cache_miss or 0),
                cache_write=int(cache_write),
                ratio=ratio,
            )
        else:
            cache_status = t("command.status.no_cache_metrics", lang)
    else:
        usage_status = t("command.status.no_run", lang)
        cache_status = t("command.status.no_run", lang)

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
        cache_status=cache_status,
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
    from session_locks import get_session_lock

    session_key = scope.session_key()
    async with get_session_lock(session_key):
        result = await force_compact(get_db(), session_key=session_key)
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
