"""History compaction for secretary v2.

Auto-compaction is delegated to the `summarization-pydantic-ai` library: its
`SummarizationProcessor` is wired into the Agent as a history_processor and,
before every run, summarizes the head of an over-threshold conversation while
keeping a recent tail. This module is now thin — it only provides:

  - build_summarization_processor(): factory used by runtime.agent
  - force_compact(): the user-triggered /compact. Unlike the in-flight
    processor (which only transforms the messages sent to the model and leaves
    the DB untouched), force_compact also rewrites the persisted snapshot to
    [summary, *tail] so subsequent runs load the compacted state directly.

Threshold decisions use a tiktoken estimate. cl100k_base is GPT's tokenizer,
not Claude's, so it's only an approximation — acceptable because the fraction
trigger keeps a margin and the processor is a safety net, not a precise
accountant. Real per-run usage is logged separately from result.usage().
"""

import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import tiktoken
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, ToolReturnPart
from pydantic_ai_summarization import (
    SummarizationProcessor,
    format_messages_for_summary,
)
from pydantic_ai_summarization.processor import (
    DEFAULT_CONTINUATION_PROMPT,
    _extract_system_prompts,
)

from config import get_config

logger = logging.getLogger(__name__)

# Sentinel the library's summarizer emits when its LLM call fails (it swallows
# the exception and returns this string as the "summary" instead of raising).
_SUMMARY_ERROR_MARKER = "Error generating summary:"

_token_encoding = None
_last_auto_persist_compact_at: Optional[float] = None


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


class SecretarySummarizationProcessor(SummarizationProcessor):
    """SummarizationProcessor with v2's shared pre/post compaction rules."""

    async def compact(
        self, messages: list[ModelMessage], reason: str = "temporary"
    ) -> CompactOutcome:
        return await _run_processor_compaction(self, messages, reason=reason)

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        outcome = await self.compact(messages, reason="temporary")
        if outcome.failed:
            logger.error("[compaction] temporary summarization failed: %s", outcome.error)
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
    """tiktoken-based token counter handed to the SummarizationProcessor.

    Reuses the library's own message formatter so counting matches what the
    summarizer actually sees.
    """
    if not messages:
        return 0
    return len(_get_encoding().encode(format_messages_for_summary(messages)))


def build_summarization_processor(force: bool = False) -> SummarizationProcessor:
    """Build the SummarizationProcessor used as a pydantic-ai history_processor.

    force=False: the agent's normal in-flight processor. Triggers at
        `compress_threshold` fraction of `context_tokens` (25% margin).
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


async def _run_processor_compaction(
    processor: SummarizationProcessor,
    messages: list[ModelMessage],
    *,
    reason: str,
) -> CompactOutcome:
    """Shared compaction implementation for temporary, manual, and auto paths."""
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
    summary_message = ModelRequest(
        parts=[*_extract_system_prompts(messages), summary_part]
    )
    compacted = [summary_message, *preserved_messages]
    after_tokens = _count_tokens(compacted)
    return CompactOutcome(
        compacted=compacted,
        changed=len(compacted) < len(messages) or after_tokens < before_tokens,
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


async def force_compact(db) -> str:
    """User-triggered /compact: summarize history and rewrite the DB snapshot.

    Archives the rolled-up rows and persists [summary, *tail] so subsequent
    runs load the compacted state directly. Returns a human-readable status.
    """
    history = db.load_pydantic_messages()
    if len(history) < 4:
        return "Not enough conversation history to compact"

    outcome = await run_compaction(history, force=True, reason="manual")
    if outcome.failed:
        logger.error("[force_compact] summarization failed: %s", outcome.error)
        return f"Compaction failed: {outcome.error}"

    if not outcome.changed:
        return "Recent history is already within budget; no compaction needed"

    try:
        archived = persist_compacted_snapshot(db, outcome)
    except Exception as e:
        logger.error(f"[force_compact] persistence failed: {e}")
        return f"Compaction partially completed; summary was generated but persistence failed: {e}"

    logger.info(
        "[force_compact] archived=%s messages=%s->%s tokens=%s->%s",
        archived,
        outcome.before_messages,
        outcome.after_messages,
        outcome.before_tokens,
        outcome.after_tokens,
    )
    return (
        f"Compaction complete: {outcome.before_messages} messages folded into "
        f"{outcome.after_messages} messages including summary"
    )


def persist_compacted_snapshot(db, outcome: CompactOutcome) -> int:
    """Archive active history and save the compacted snapshot."""
    if outcome.failed or not outcome.changed:
        return 0
    archived = db.archive_all_pydantic_messages()
    db.save_pydantic_messages(outcome.compacted)
    return archived


async def maybe_auto_persist_compact(db) -> Optional[CompactOutcome]:
    """Persist a compacted snapshot when active history is sustainably large."""
    global _last_auto_persist_compact_at

    cfg = get_config().history
    if not cfg.auto_persist_compact:
        return None

    now = time.monotonic()
    cooldown_sec = cfg.persist_compact_cooldown_minutes * 60
    if (
        _last_auto_persist_compact_at is not None
        and now - _last_auto_persist_compact_at < cooldown_sec
    ):
        return None

    history = db.load_pydantic_messages()
    if len(history) < cfg.persist_compact_min_active_messages:
        return None

    tokens = _count_tokens(history)
    threshold = int(cfg.context_tokens * cfg.persist_compact_threshold)
    if tokens < threshold:
        return None

    outcome = await run_compaction(history, force=True, reason="auto_persist")
    if outcome.failed:
        logger.error("[auto_compact] summarization failed: %s", outcome.error)
        return outcome
    if not outcome.changed:
        return outcome

    archived = persist_compacted_snapshot(db, outcome)
    _last_auto_persist_compact_at = now
    logger.info(
        "[auto_compact] archived=%s messages=%s->%s tokens=%s->%s",
        archived,
        outcome.before_messages,
        outcome.after_messages,
        outcome.before_tokens,
        outcome.after_tokens,
    )
    return outcome
