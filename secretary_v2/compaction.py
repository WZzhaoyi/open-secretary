"""History compaction for secretary v2.

Before each run, runtime asks this module whether the active persisted history
has crossed the configured compaction threshold. If it has, the head is
summarized, the recent tail is kept verbatim, and the compacted snapshot is
written back to the DB before the model call. Manual /compact uses the same
snapshot rewrite path.

Threshold decisions use a tiktoken estimate. cl100k_base is GPT's tokenizer,
not Claude's, so it's only an approximation — acceptable because the fraction
trigger keeps a margin and the processor is a safety net, not a precise
accountant. Real per-run usage is logged separately from result.usage().
"""

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, Literal, Optional
from zoneinfo import ZoneInfo

import tiktoken
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
    ToolReturnPart,
)
from pydantic_ai_summarization import (
    SummarizationProcessor,
    format_messages_for_summary,
)
from pydantic_ai_summarization.processor import (
    DEFAULT_CONTINUATION_PROMPT,
)

from config import get_config

logger = logging.getLogger(__name__)

# Sentinel the library's summarizer emits when its LLM call fails (it swallows
# the exception and returns this string as the "summary" instead of raising).
_SUMMARY_ERROR_MARKER = "Error generating summary:"

_token_encoding = None
_last_auto_persist_compact_at: Dict[str, float] = {}
_auto_compact_failures: Dict[str, "_FailureBackoff"] = {}
_FAILURE_BACKOFF_BASE_SECONDS = 60
_FAILURE_BACKOFF_MAX_SECONDS = 15 * 60


@dataclass(frozen=True)
class _FailureBackoff:
    attempts: int
    retry_at: float


@dataclass
class CompactOutcome:
    """Result of a compaction attempt, independent from persistence."""

    compacted: list[ModelMessage]
    changed: bool
    failed: bool
    error: Optional[str]
    before_messages: int
    after_messages: int
    before_tokens: int
    after_tokens: int
    reason: str


@dataclass(frozen=True)
class CompactCommandResult:
    """Structured result for channel-independent /compact presentation."""

    status: Literal[
        "completed",
        "not_needed",
        "not_enough_history",
        "failed",
        "partial_failure",
    ]
    before_messages: int = 0
    after_messages: int = 0
    error: Optional[str] = None


class SecretarySummarizationProcessor(SummarizationProcessor):
    """SummarizationProcessor with v2's shared pre/post compaction rules."""

    async def compact(
        self, messages: list[ModelMessage], reason: str = "auto"
    ) -> CompactOutcome:
        return await _run_processor_compaction(self, messages, reason=reason)

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        outcome = await self.compact(messages, reason="auto")
        if outcome.failed:
            logger.error("[compaction] summarization failed: %s", outcome.error)
            return messages
        return outcome.compacted


def _get_encoding():
    global _token_encoding
    if _token_encoding is None:
        _token_encoding = tiktoken.get_encoding("cl100k_base")
    return _token_encoding


def estimate_tokens(text: str) -> int:
    """Estimate token count using cl100k_base (close enough for budget decisions)."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def _count_tokens(messages) -> int:
    """Estimate the complete request history without hiding large tool output.

    The summarization library's formatter intentionally truncates tool returns
    for its own prompt. It therefore cannot be used for context-window or
    trigger decisions, which must account for the full provider request.
    """
    if not messages:
        return 0
    serialized = bytes(ModelMessagesTypeAdapter.dump_json(messages)).decode(
        "utf-8", errors="replace"
    )
    return len(_get_encoding().encode(serialized))


def build_summarization_processor(force: bool = False) -> SummarizationProcessor:
    """Build the SummarizationProcessor used by automatic and manual compaction.

    force=False: automatic pre-run compaction. Triggers at `compress_threshold`
        fraction of `context_tokens`.
    force=True: used by /compact — triggers on any non-trivial history so the
        user can compact on demand regardless of current size.
    """
    # Lazy import to break the runtime <-> compaction circular import.
    from runtime import build_model

    cfg = get_config().history
    trigger = ("messages", 4) if force else ("fraction", cfg.compress_threshold)

    return SecretarySummarizationProcessor(
        model=build_model(get_config()),
        trigger=trigger,
        keep=("tokens", cfg.tail_token_budget),
        token_counter=_count_tokens,
        max_input_tokens=cfg.context_tokens,
    )


def _truncate_tool_output(content: str, max_chars: int) -> str:
    if max_chars <= 0 or len(content) <= max_chars:
        return content

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        f"[Tool output truncated for compaction: original {len(content)} chars, "
        f"kept head {head_chars} + tail {tail_chars}]\n"
        f"{content[:head_chars]}\n"
        "[... truncated ...]\n"
        f"{content[-tail_chars:]}"
    )


def _prune_tool_outputs_for_summary(
    messages: list[ModelMessage], max_chars: int
) -> list[ModelMessage]:
    """Return a copy of head messages with oversized tool returns shortened.

    We only alter the messages fed into the summarizer. The preserved tail is
    kept verbatim, and tool return parts are replaced rather than removed so
    tool-call/tool-return structure remains valid while summarizing.
    """
    if max_chars <= 0:
        return messages

    pruned: list[ModelMessage] = []
    for msg in messages:
        changed = False
        new_parts = []
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolReturnPart):
                content = part.content
                text = content if isinstance(content, str) else str(content)
                if len(text) > max_chars:
                    part = replace(part, content=_truncate_tool_output(text, max_chars))
                    changed = True
            new_parts.append(part)
        pruned.append(replace(msg, parts=new_parts) if changed else msg)
    return pruned


def _temporal_anchoring_prompt() -> str:
    """Instruction injected only into the summarizer input.

    The live agent gets the authoritative clock in the runtime tail. Compaction
    needs a separate guard because summaries can persist for a long time and be
    replayed as history.
    """
    cfg = get_config()
    try:
        today = datetime.now(ZoneInfo(cfg.timezone)).date().isoformat()
    except Exception:
        today = datetime.now().date().isoformat()
    return (
        "TEMPORAL ANCHORING FOR SUMMARY ONLY:\n"
        f"- The current local date is {today} in timezone {cfg.timezone}.\n"
        "- Rewrite completed actions as absolute, dated, past-tense facts.\n"
        "- Do not preserve old relative phrases such as today/tomorrow/tonight "
        "as if they were current. Resolve them to historical facts when the "
        "source text makes that possible.\n"
        "- Never invent a date for work that has not happened yet. Keep future "
        "reminders explicitly future-dated."
    )


def _add_temporal_anchoring_for_summary(
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[SystemPromptPart(content=_temporal_anchoring_prompt())]),
        *messages,
    ]


async def _run_processor_compaction(
    processor: SummarizationProcessor,
    messages: list[ModelMessage],
    *,
    reason: str,
) -> CompactOutcome:
    """Shared compaction implementation for manual and auto paths."""
    before_tokens = _count_tokens(messages)
    unchanged = CompactOutcome(
        compacted=messages,
        changed=False,
        failed=False,
        error=None,
        before_messages=len(messages),
        after_messages=len(messages),
        before_tokens=before_tokens,
        after_tokens=before_tokens,
        reason=reason,
    )

    if not messages:
        return unchanged

    if not processor._should_summarize(messages, before_tokens):
        return unchanged

    cutoff_index = processor._determine_cutoff_index(messages)
    if cutoff_index <= 0:
        return unchanged

    cfg = get_config().history
    messages_to_summarize = _prune_tool_outputs_for_summary(
        messages[:cutoff_index], cfg.compact_tool_output_max_chars
    )
    messages_to_summarize = _add_temporal_anchoring_for_summary(messages_to_summarize)
    preserved_messages = messages[cutoff_index:]

    try:
        summary = await processor._create_summary(messages_to_summarize)
    except Exception as e:
        return CompactOutcome(
            compacted=messages,
            changed=False,
            failed=True,
            error=str(e),
            before_messages=len(messages),
            after_messages=len(messages),
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            reason=reason,
        )

    if _SUMMARY_ERROR_MARKER in summary:
        return CompactOutcome(
            compacted=messages,
            changed=False,
            failed=True,
            error=summary,
            before_messages=len(messages),
            after_messages=len(messages),
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            reason=reason,
        )

    summary_part = SystemPromptPart(content=f"{DEFAULT_CONTINUATION_PROMPT}{summary}")
    # App-managed system/runtime prompts are rebuilt for every run. Persisting
    # extracted prompts here previously froze old clocks and reintroduced them
    # after every compaction.
    summary_message = ModelRequest(parts=[summary_part])
    compacted = [summary_message, *preserved_messages]
    after_tokens = _count_tokens(compacted)
    target_tokens = int(cfg.context_tokens * cfg.compress_threshold)
    if after_tokens >= before_tokens:
        error = (
            "compaction did not reduce token usage "
            f"({before_tokens} -> {after_tokens})"
        )
    elif reason == "auto" and after_tokens >= target_tokens:
        error = (
            "automatic compaction did not reach its target "
            f"({after_tokens} >= {target_tokens} tokens)"
        )
    else:
        error = None
    if error:
        return CompactOutcome(
            compacted=messages,
            changed=False,
            failed=True,
            error=error,
            before_messages=len(messages),
            after_messages=len(messages),
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            reason=reason,
        )
    return CompactOutcome(
        compacted=compacted,
        changed=True,
        failed=False,
        error=None,
        before_messages=len(messages),
        after_messages=len(compacted),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        reason=reason,
    )


async def run_compaction(
    messages: list[ModelMessage],
    *,
    force: bool = False,
    reason: str = "manual",
) -> CompactOutcome:
    processor = build_summarization_processor(force=force)
    if isinstance(processor, SecretarySummarizationProcessor):
        return await processor.compact(messages, reason=reason)
    return await _run_processor_compaction(processor, messages, reason=reason)


def _summary_failed(compacted) -> bool:
    """True if the library's summarizer fell back to its error sentinel.

    The library never raises on summarizer failure; it embeds the error text
    as the summary. We must detect that so force_compact doesn't persist a
    broken snapshot (per the original critique: never degrade to a placeholder).
    """
    if not compacted:
        return False
    for part in getattr(compacted[0], "parts", []):
        content = getattr(part, "content", "")
        if isinstance(content, str) and _SUMMARY_ERROR_MARKER in content:
            return True
    return False


async def force_compact(
    db,
    session_key: Optional[str] = None,
    created_after: Optional[datetime] = None,
    include_legacy: bool = True,
) -> CompactCommandResult:
    """User-triggered /compact: summarize history and rewrite the DB snapshot.

    Archives the rolled-up rows and persists [summary, *tail] so subsequent
    runs load the compacted state directly. Presentation belongs to the
    channel-independent command layer, so this returns structured data.
    """
    loaded_row_ids: list[int] = []
    history = db.load_pydantic_messages(
        session_key=session_key,
        include_legacy=include_legacy,
        created_after=created_after,
        loaded_row_ids=loaded_row_ids,
    )
    if len(history) < 4:
        return CompactCommandResult(status="not_enough_history")

    outcome = await run_compaction(history, force=True, reason="manual")
    if outcome.failed:
        logger.error("[force_compact] summarization failed: %s", outcome.error)
        return CompactCommandResult(status="failed", error=outcome.error)

    if not outcome.changed:
        return CompactCommandResult(status="not_needed")

    try:
        archived = persist_compacted_snapshot(
            db,
            outcome,
            session_key=session_key,
            archive_row_ids=loaded_row_ids,
        )
    except Exception as e:
        logger.error(f"[force_compact] persistence failed: {e}")
        return CompactCommandResult(status="partial_failure", error=str(e))

    logger.info(
        "[force_compact] archived=%s messages=%s->%s tokens=%s->%s",
        archived,
        outcome.before_messages,
        outcome.after_messages,
        outcome.before_tokens,
        outcome.after_tokens,
    )
    _auto_compact_failures.pop(session_key or "__legacy__", None)
    return CompactCommandResult(
        status="completed",
        before_messages=outcome.before_messages,
        after_messages=outcome.after_messages,
    )


def persist_compacted_snapshot(
    db,
    outcome: CompactOutcome,
    session_key: Optional[str] = None,
    archive_row_ids: Optional[list[int]] = None,
) -> int:
    """Archive active history and save the compacted snapshot."""
    if outcome.failed or not outcome.changed:
        return 0
    if archive_row_ids is None:
        raise ValueError("compaction persistence requires loaded row provenance")
    return db.replace_pydantic_messages_snapshot(
        outcome.compacted,
        archive_row_ids=archive_row_ids,
        session_key=session_key,
    )


async def maybe_auto_persist_compact(
    db,
    history: Optional[list[ModelMessage]] = None,
    session_key: Optional[str] = None,
    loaded_row_ids: Optional[list[int]] = None,
) -> Optional[CompactOutcome]:
    """Persist a compacted snapshot when active history crosses one threshold."""
    global _last_auto_persist_compact_at, _auto_compact_failures

    cfg = get_config().history
    if not cfg.auto_compact:
        return None

    now = time.monotonic()
    cooldown_sec = cfg.compact_cooldown_minutes * 60
    cooldown_key = session_key or "__legacy__"
    if cooldown_sec > 0:
        expired_keys = [
            key
            for key, compacted_at in _last_auto_persist_compact_at.items()
            if now - compacted_at >= cooldown_sec
        ]
        for key in expired_keys:
            _last_auto_persist_compact_at.pop(key, None)
    last_compact_at = _last_auto_persist_compact_at.get(cooldown_key)

    if history is None:
        if loaded_row_ids is None:
            loaded_row_ids = []
        history = db.load_pydantic_messages(
            session_key=session_key,
            loaded_row_ids=loaded_row_ids,
        )
    if len(history) < cfg.compact_min_active_messages:
        return None

    tokens = _count_tokens(history)
    threshold = int(cfg.context_tokens * cfg.compress_threshold)
    if tokens < threshold:
        _auto_compact_failures.pop(cooldown_key, None)
        return None

    emergency_threshold = int(cfg.context_tokens * 0.9)
    if last_compact_at is not None and now - last_compact_at < cooldown_sec:
        if tokens < emergency_threshold:
            remaining = max(0, cooldown_sec - (now - last_compact_at))
            logger.debug(
                "[auto_compact] cooldown skip session=%s remaining=%.0fs",
                cooldown_key,
                remaining,
            )
            return None
        logger.warning(
            "[auto_compact] bypassing cooldown for near-limit session=%s tokens=%s",
            cooldown_key,
            tokens,
        )

    failure = _auto_compact_failures.get(cooldown_key)
    if failure is not None and now < failure.retry_at:
        logger.debug(
            "[auto_compact] failure backoff skip session=%s remaining=%.0fs",
            cooldown_key,
            failure.retry_at - now,
        )
        return None

    def record_failure(error: str) -> None:
        previous = _auto_compact_failures.get(cooldown_key)
        attempts = (previous.attempts if previous else 0) + 1
        delay = min(
            _FAILURE_BACKOFF_BASE_SECONDS * (2 ** min(attempts - 1, 4)),
            _FAILURE_BACKOFF_MAX_SECONDS,
        )
        _auto_compact_failures[cooldown_key] = _FailureBackoff(
            attempts=attempts,
            retry_at=now + delay,
        )
        logger.warning(
            "[auto_compact] failure backoff session=%s attempts=%s retry_in=%ss error=%s",
            cooldown_key,
            attempts,
            delay,
            error,
        )

    outcome = await run_compaction(history, force=True, reason="auto")
    if outcome.failed:
        logger.error("[auto_compact] summarization failed: %s", outcome.error)
        record_failure(outcome.error or "unknown compaction failure")
        return outcome
    if not outcome.changed:
        outcome = replace(
            outcome,
            failed=True,
            error="automatic compaction did not reduce over-budget history",
        )
        record_failure(outcome.error)
        return outcome

    try:
        archived = persist_compacted_snapshot(
            db,
            outcome,
            session_key=session_key,
            archive_row_ids=loaded_row_ids,
        )
    except Exception as e:
        failed_outcome = replace(
            outcome,
            compacted=history,
            changed=False,
            failed=True,
            error=f"compaction persistence failed: {e}",
            after_messages=len(history),
            after_tokens=tokens,
        )
        record_failure(failed_outcome.error or str(e))
        return failed_outcome
    _auto_compact_failures.pop(cooldown_key, None)
    if cooldown_sec > 0:
        _last_auto_persist_compact_at[cooldown_key] = now
    else:
        _last_auto_persist_compact_at.pop(cooldown_key, None)
    logger.info(
        "[auto_compact] archived=%s messages=%s->%s tokens=%s->%s",
        archived,
        outcome.before_messages,
        outcome.after_messages,
        outcome.before_tokens,
        outcome.after_tokens,
    )
    return outcome
