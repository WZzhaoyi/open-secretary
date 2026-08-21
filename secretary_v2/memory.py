"""Secretary v2 memory module - SQLAlchemy based."""

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def _utcnow():
    """Naive UTC datetime — replaces datetime.utcnow() (deprecated in 3.12+)
    while keeping the existing schema semantics (naive datetime columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _event_summary(content: str, summary: Optional[str] = None, max_chars: int = 160) -> str:
    """Short event index text stored separately from full content."""
    text = summary if summary is not None and str(summary).strip() else content
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"

from sqlalchemy import (
    BLOB,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

import tiktoken
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ThinkingPart,
    UserPromptPart,
)

from config import get_config

logger = logging.getLogger(__name__)

Base = declarative_base()

_token_encoding = None
_SUMMARY_MARKER = "Summary of previous conversation:"


def _estimate_msg_tokens(msgs: List[ModelMessage]) -> int:
    """Rough tiktoken estimate for a batch of messages, used only as a load
    boundary in load_pydantic_messages. cl100k_base is GPT's tokenizer, not
    Claude's — precision doesn't matter here; the SummarizationProcessor is
    the real safety net."""
    global _token_encoding
    if _token_encoding is None:
        _token_encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in msgs:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += len(_token_encoding.encode(content))
            elif content is not None:
                total += len(_token_encoding.encode(str(content)))
    return total


def _trim_to_complete_turn(messages: List[ModelMessage]) -> List[ModelMessage]:
    """Drop a time-window fragment until a replay-safe conversation boundary.

    Rows from one agent run are persisted separately. A timestamp cutoff can
    therefore land between the initial user request and its response/tool loop.
    Start at the next real user request, while allowing a compacted summary to
    remain the first replayed message.
    """
    for index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            continue
        if any(isinstance(part, UserPromptPart) for part in message.parts):
            return messages[index:]
        if any(
            isinstance(part, SystemPromptPart)
            and _SUMMARY_MARKER in part.content
            for part in message.parts
        ):
            return messages[index:]
    return []


def _response_has_actionable_part(msg: ModelResponse) -> bool:
    """True when a response can be replayed as assistant content/tool_calls."""
    for part in msg.parts:
        kind = getattr(part, "part_kind", "")
        if kind == "text" and bool(getattr(part, "content", None)):
            return True
        if kind == "tool-call":
            return True
    return False


def _sanitize_response_for_history(msg: ModelResponse) -> Optional[ModelResponse]:
    """Keep usable reasoning while rejecting responses that cannot be replayed."""
    parts = [
        part
        for part in msg.parts
        if not isinstance(part, ThinkingPart) or bool(part.content.strip())
    ]
    if not parts:
        return None
    cleaned = msg if len(parts) == len(msg.parts) else replace(msg, parts=parts)
    if not _response_has_actionable_part(cleaned):
        return None
    return cleaned


def _tool_call_ids(msg: ModelResponse) -> List[str]:
    return [
        tool_call_id
        for part in msg.parts
        if getattr(part, "part_kind", "") == "tool-call"
        for tool_call_id in [getattr(part, "tool_call_id", None)]
        if tool_call_id
    ]


def _tool_return_ids(msg: ModelRequest) -> List[str]:
    return [
        tool_call_id
        for part in msg.parts
        if getattr(part, "part_kind", "") in {"tool-return", "retry-prompt"}
        for tool_call_id in [getattr(part, "tool_call_id", None)]
        if tool_call_id
    ]


def _strip_tool_returns(msg: ModelRequest) -> Optional[ModelRequest]:
    parts = [
        part
        for part in msg.parts
        if getattr(part, "part_kind", "") not in {"tool-return", "retry-prompt"}
    ]
    if not parts:
        return None
    return msg if len(parts) == len(msg.parts) else replace(msg, parts=parts)


def _drop_incomplete_tool_call_sequences(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Drop history fragments that violate OpenAI tool-call adjacency rules."""
    result: List[ModelMessage] = []
    pending_ids: set[str] = set()
    pending_messages: List[ModelMessage] = []

    def drop_pending(reason: str) -> None:
        if pending_messages:
            logger.warning(
                "Dropping incomplete tool-call history (%s): pending_tool_call_ids=%s",
                reason,
                sorted(pending_ids),
            )
        pending_ids.clear()
        pending_messages.clear()

    for msg in messages:
        if pending_ids:
            if isinstance(msg, ModelRequest):
                returns = set(_tool_return_ids(msg))
                if returns:
                    pending_messages.append(msg)
                    pending_ids.difference_update(returns)
                    if not pending_ids:
                        result.extend(pending_messages)
                        pending_messages.clear()
                    continue

                drop_pending("next request had no tool return")
                stripped = _strip_tool_returns(msg)
                if stripped is not None:
                    result.append(stripped)
                continue

            drop_pending("next message was not a tool return")

        if isinstance(msg, ModelResponse):
            call_ids = set(_tool_call_ids(msg))
            if call_ids:
                pending_ids = call_ids
                pending_messages = [msg]
            else:
                result.append(msg)
            continue

        if isinstance(msg, ModelRequest):
            stripped = _strip_tool_returns(msg)
            if stripped is not None:
                result.append(stripped)
            elif _tool_return_ids(msg):
                logger.warning("Dropping orphan tool-return history message")
            continue

        result.append(msg)

    drop_pending("end of history")
    return result


def _strip_non_replayable_messages(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    sanitized: List[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            cleaned = _sanitize_response_for_history(msg)
            if cleaned is None:
                logger.warning(
                    "Skipping non-actionable model response with parts=%s",
                    [getattr(part, "part_kind", type(part).__name__) for part in msg.parts],
                )
                continue
            sanitized.append(cleaned)
        else:
            sanitized.append(msg)
    return sanitized


def sanitize_pydantic_messages_for_history(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Return only model messages safe to persist and replay as history.

    Assistant responses with neither visible content nor tool_calls cannot be
    replayed safely, so thinking-only responses are rejected. Non-empty reasoning
    is retained when the same response has visible text or tool_calls; content=None
    remains valid when tool_calls are present.

    OpenAI-compatible history also requires every assistant tool-call response
    to be followed by tool-return messages for all tool_call_id values before
    the next assistant response. Legacy SQLite rows can lose one side of that
    pair, so we drop incomplete fragments instead of replaying invalid history.
    """
    sanitized = _strip_non_replayable_messages(messages)
    return _drop_incomplete_tool_call_sequences(sanitized)


class Event(Base):
    """事件记录表 - Compatible with original schema."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)  # remind | check | response | note | triggered
    status = Column(String, nullable=False, default="logged", server_default="logged")
    summary = Column(Text, nullable=True)
    content = Column(Text)
    created_at = Column(DateTime, default=_utcnow)
    source_channel = Column(String, nullable=True)
    session_key = Column(String, nullable=True, index=True)
    source_message_id = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)
    context_visible = Column(
        Integer, nullable=False, default=1, server_default="1", index=True
    )


class Message(Base):
    """对话历史表 - Compatible with original schema."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)  # user/assistant/system/summary
    content = Column(Text)
    tool_calls = Column(Text)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
    pydantic_ai_msg = Column(BLOB, nullable=True)  # 序列化的 Pydantic AI 消息
    agent_id = Column(String, nullable=True, default="secretary", index=True)
    session_key = Column(String, nullable=True, index=True)
    channel = Column(String, nullable=True)
    conversation_id = Column(String, nullable=True)
    thread_id = Column(String, nullable=True)
    sender_id = Column(String, nullable=True)
    reply_to_id = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)
    context_visible = Column(
        Integer, nullable=False, default=1, server_default="1", index=True
    )


class ScheduledTask(Base):
    """定时任务表 - Compatible with original schema."""
    __tablename__ = "scheduled_tasks"

    id = Column(String, primary_key=True)
    cron = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    enabled = Column(Integer, default=1)
    protected = Column(Integer, default=0)
    handler = Column(String, nullable=False, default="agent", server_default="agent")
    builtin_task = Column(String, nullable=True)
    last_attempt = Column(DateTime, nullable=True)
    last_success = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    last_run = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class SubAgentRun(Base):
    """Generic persisted background subagent run."""
    __tablename__ = "subagent_runs"

    id = Column(String, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    agent_kind = Column(String, nullable=False, default="", index=True)
    engine = Column(String, nullable=False)
    input_json = Column(Text, nullable=False, default="{}")
    subject = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="pending", index=True)
    origin_channel = Column(String, nullable=False, default="cli")
    user_id = Column(String, nullable=True)
    stages_json = Column(Text, default="[]")
    artifact_path = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

    @property
    def input_payload(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.input_json or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

class AgentEvent(Base):
    """Agent behavior audit events.

    This is not long-term memory and is not injected into prompts. It is a
    local observability trail for answering operational questions such as
    whether a scheduled run fired, sent a message, updated memory, or stayed
    silent intentionally.
    """
    __tablename__ = "agent_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=True, index=True)
    origin = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    subject = Column(String, nullable=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, index=True)


class Database:
    """Database manager."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            config = get_config()
            db_path = config.database.path

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        Base.metadata.create_all(self.engine)
        self._ensure_event_status_column()
        self._ensure_columns(
            "events",
            {
                "summary": "TEXT",
                "source_channel": "TEXT",
                "session_key": "TEXT",
                "source_message_id": "TEXT",
                "metadata_json": "TEXT",
                "context_visible": "INTEGER NOT NULL DEFAULT 1",
            },
        )
        self._ensure_columns(
            "messages",
            {
                "agent_id": "TEXT DEFAULT 'secretary'",
                "session_key": "TEXT",
                "channel": "TEXT",
                "conversation_id": "TEXT",
                "thread_id": "TEXT",
                "sender_id": "TEXT",
                "reply_to_id": "TEXT",
                "metadata_json": "TEXT",
                "context_visible": "INTEGER NOT NULL DEFAULT 1",
            },
        )
        self._ensure_columns(
            "scheduled_tasks",
            {
                "handler": "TEXT NOT NULL DEFAULT 'agent'",
                "builtin_task": "TEXT",
                "last_attempt": "DATETIME",
                "last_success": "DATETIME",
                "last_error": "TEXT",
            },
        )
        self._backfill_event_summaries()

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        """Add missing SQLite columns for existing local databases."""
        with self.engine.begin() as conn:
            existing = {
                row._mapping["name"]
                for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            for name, ddl_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))

    def _ensure_event_status_column(self) -> None:
        """Backfill schema for existing SQLite databases.

        SQLAlchemy create_all() creates missing tables but does not alter existing
        ones. Older secretary_v2.db files need the events.status column added
        explicitly so direct db_execute INSERTs keep working.
        """
        with self.engine.begin() as conn:
            columns = {
                row._mapping["name"]
                for row in conn.execute(text("PRAGMA table_info(events)"))
            }
            if "status" not in columns:
                conn.execute(
                    text(
                        "ALTER TABLE events "
                        "ADD COLUMN status TEXT NOT NULL DEFAULT 'logged'"
                    )
                )

    def _backfill_event_summaries(self) -> None:
        """Populate event summaries for rows created before the summary column."""
        with self.get_session() as session:
            rows = (
                session.query(Event)
                .filter((Event.summary.is_(None)) | (Event.summary == ""))
                .all()
            )
            if not rows:
                return
            for event in rows:
                event.summary = _event_summary(event.content or "")
            session.commit()

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    # Event operations
    def create_event(
        self,
        event_type: str,
        content: str,
        status: str = "logged",
        *,
        summary: Optional[str] = None,
        source_channel: Optional[str] = None,
        session_key: Optional[str] = None,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Create a new event."""
        metadata_json = (
            json.dumps(metadata, ensure_ascii=False, default=str)
            if metadata is not None
            else None
        )
        with self.get_session() as session:
            event = Event(
                type=event_type,
                status=status,
                summary=_event_summary(content, summary),
                content=content,
                source_channel=source_channel,
                session_key=session_key,
                source_message_id=source_message_id,
                metadata_json=metadata_json,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def update_event_summary(self, event_id: int, summary: str) -> Optional[Event]:
        """Update only the short index summary for an event."""
        with self.get_session() as session:
            event = session.get(Event, event_id)
            if event is None:
                return None
            event.summary = _event_summary(event.content or "", summary)
            session.commit()
            session.refresh(event)
            return event

    def get_events(self, limit: int = 50) -> List[Event]:
        """Get recent events."""
        with self.get_session() as session:
            return session.query(Event).order_by(
                Event.created_at.desc()
            ).limit(limit).all()

    def get_events_excluding_statuses(
        self, excluded_statuses: List[str], limit: int = 50
    ) -> List[Event]:
        """Get context-visible recent events while omitting status buckets."""
        with self.get_session() as session:
            query = session.query(Event).filter(Event.context_visible == 1)
            if excluded_statuses:
                query = query.filter(Event.status.notin_(excluded_statuses))
            return query.order_by(Event.created_at.desc()).limit(limit).all()

    def get_events_by_status(self, status: str, limit: int = 200) -> List[Event]:
        """Get context-visible events by attention status, newest first."""
        with self.get_session() as session:
            return (
                session.query(Event)
                .filter(Event.status == status)
                .filter(Event.context_visible == 1)
                .order_by(Event.created_at.desc())
                .limit(limit)
                .all()
            )

    def count_events_by_status(self, status: str) -> int:
        """Count context-visible events by attention status."""
        with self.get_session() as session:
            return (
                session.query(Event)
                .filter(Event.status == status)
                .filter(Event.context_visible == 1)
                .count()
            )

    # Message operations
    def save_message(self, source: str, content: str,
                     tokens_in: int = 0, tokens_out: int = 0,
                     tool_calls: Optional[str] = None,
                     pydantic_ai_msg: Optional[bytes] = None) -> Message:
        """Save a message to history."""
        with self.get_session() as session:
            message = Message(
                source=source,
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=tool_calls,
                pydantic_ai_msg=pydantic_ai_msg,
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message

    def get_messages(self, limit: int = 100) -> List[Message]:
        """Get recent messages."""
        with self.get_session() as session:
            return session.query(Message).order_by(
                Message.created_at.desc()
            ).limit(limit).all()

    def save_pydantic_messages(
        self,
        messages: List[ModelMessage],
        preview_text: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        session_key: Optional[str] = None,
        channel: Optional[str] = None,
        conversation_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        reply_to_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a batch of Pydantic AI messages from one agent run.

        Each ModelMessage is stored as one row, in order, so load_pydantic_messages
        can reconstruct the conversation. The pydantic_ai_msg BLOB column holds
        the canonical serialized form; content holds a short text preview for humans.
        """
        messages = sanitize_pydantic_messages_for_history(messages)
        if not messages:
            return
        metadata_json = (
            json.dumps(metadata, ensure_ascii=False, default=str)
            if metadata is not None
            else None
        )
        with self.get_session() as session:
            for msg in messages:
                blob = bytes(ModelMessagesTypeAdapter.dump_json([msg]))
                source = "request" if msg.kind == "request" else "response"
                preview = preview_text or _preview_for_message(msg)
                session.add(
                    Message(
                        source=source,
                        content=preview[:500] if preview else "",
                        pydantic_ai_msg=blob,
                        agent_id=agent_id,
                        session_key=session_key,
                        channel=channel,
                        conversation_id=conversation_id,
                        thread_id=thread_id,
                        sender_id=sender_id,
                        reply_to_id=reply_to_id,
                        metadata_json=metadata_json,
                    )
                )
            session.commit()

    def load_pydantic_messages(
        self,
        token_budget: Optional[int] = None,
        batch_size: int = 100,
        *,
        session_key: Optional[str] = None,
        include_legacy: bool = True,
        created_after: Optional[datetime] = None,
        loaded_row_ids: Optional[List[int]] = None,
    ) -> List[ModelMessage]:
        """Load recent pydantic-ai messages newest-first up to a token budget.

        There is no fixed message-count cap: depth emerges from the token
        budget (defaults to history.context_tokens). Rows are walked
        newest-first; once the budget is exceeded we stop, so the oldest rows
        beyond the budget are simply not loaded. The newest row is always
        included even if it alone exceeds the budget.

        Pre-run compaction is the real safety net — this budget only bounds how
        much we deserialize and feed it. ``created_after`` is a read-time soft
        retention boundary; older rows remain stored. Rows with
        ``context_visible=0`` or without a pydantic_ai_msg payload (legacy
        plain-text rows, or rows archived by /compact) are skipped.
        """
        if token_budget is None:
            token_budget = get_config().history.context_tokens

        selected: List[tuple[int, List[ModelMessage]]] = []
        total = 0
        offset = 0
        visibility_boundary = False

        with self.get_session() as session:
            hidden_query = session.query(Message.id).filter(
                Message.context_visible == 0
            )
            if session_key is not None:
                if include_legacy:
                    hidden_query = hidden_query.filter(
                        (Message.session_key == session_key)
                        | (Message.session_key.is_(None))
                    )
                else:
                    hidden_query = hidden_query.filter(
                        Message.session_key == session_key
                    )
            visibility_boundary = hidden_query.first() is not None

            while True:
                query = session.query(Message.id, Message.pydantic_ai_msg).filter(
                    Message.pydantic_ai_msg.isnot(None),
                    Message.context_visible == 1,
                )
                if created_after is not None:
                    query = query.filter(Message.created_at >= created_after)
                if session_key is not None:
                    if include_legacy:
                        query = query.filter(
                            (Message.session_key == session_key)
                            | (Message.session_key.is_(None))
                        )
                    else:
                        query = query.filter(Message.session_key == session_key)
                rows = (
                    query.order_by(Message.id.desc())
                    .offset(offset)
                    .limit(batch_size)
                    .all()
                )
                if not rows:
                    break

                should_stop = False
                for row in rows:
                    try:
                        msgs = ModelMessagesTypeAdapter.validate_json(row.pydantic_ai_msg)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize message id={row.id}: {e}")
                        continue
                    msgs = _strip_non_replayable_messages(msgs)
                    if not msgs:
                        logger.warning(
                            "Skipping non-actionable serialized message id=%s",
                            row.id,
                        )
                        continue
                    cost = _estimate_msg_tokens(msgs)
                    if selected and total + cost > token_budget:
                        should_stop = True
                        break
                    selected.append((row.id, msgs))
                    total += cost

                if should_stop or len(rows) < batch_size:
                    break
                offset += batch_size

        selected.reverse()  # back to chronological order
        result: List[ModelMessage] = []
        for _, msgs in selected:
            result.extend(msgs)
        # A time cutoff or context-visibility update can hide only the first
        # part of a persisted run. Advance to a replay-safe request boundary
        # without changing legacy response-only histories that have no cutoff.
        if created_after is not None or visibility_boundary:
            result = _trim_to_complete_turn(result)
        if loaded_row_ids is not None:
            retained_message_ids = {id(message) for message in result}
            loaded_row_ids.extend(
                row_id
                for row_id, msgs in selected
                if any(id(message) in retained_message_ids for message in msgs)
            )
        return sanitize_pydantic_messages_for_history(result)

    def replace_pydantic_messages_snapshot(
        self,
        messages: List[ModelMessage],
        *,
        archive_row_ids: List[int],
        session_key: Optional[str] = None,
    ) -> int:
        """Atomically archive selected rows and insert their compacted snapshot."""
        messages = sanitize_pydantic_messages_for_history(messages)
        if not archive_row_ids or not messages:
            return 0

        with self.get_session() as session:
            try:
                archived = (
                    session.query(Message)
                    .filter(Message.id.in_(archive_row_ids))
                    .filter(Message.pydantic_ai_msg.isnot(None))
                    .update({Message.pydantic_ai_msg: None}, synchronize_session=False)
                )
                for msg in messages:
                    blob = bytes(ModelMessagesTypeAdapter.dump_json([msg]))
                    session.add(
                        Message(
                            source="request" if msg.kind == "request" else "response",
                            content=_preview_for_message(msg)[:500],
                            pydantic_ai_msg=blob,
                            session_key=session_key,
                        )
                    )
                session.commit()
                return archived
            except Exception:
                session.rollback()
                raise

    def archive_invalid_pydantic_messages(self) -> int:
        """Sanitize active history blobs and archive rows that cannot be replayed.

        Rows that become empty after sanitization have their BLOB nulled so
        load_pydantic_messages skips them while the human-readable audit preview
        remains in `content`. Rows that only needed empty thinking parts removed
        are rewritten with the sanitized blob.
        """
        changed = 0
        with self.get_session() as session:
            rows = (
                session.query(Message)
                .filter(Message.pydantic_ai_msg.isnot(None))
                .order_by(Message.id.asc())
                .all()
            )
            for row in rows:
                try:
                    msgs = ModelMessagesTypeAdapter.validate_json(row.pydantic_ai_msg)
                except Exception as e:
                    logger.warning(f"Archiving unreadable message id={row.id}: {e}")
                    row.pydantic_ai_msg = None
                    changed += 1
                    continue

                sanitized = _strip_non_replayable_messages(msgs)
                if not sanitized:
                    row.pydantic_ai_msg = None
                    changed += 1
                    continue

                new_blob = bytes(ModelMessagesTypeAdapter.dump_json(sanitized))
                if new_blob != row.pydantic_ai_msg:
                    row.pydantic_ai_msg = new_blob
                    changed += 1
            session.commit()
        return changed

    def archive_pydantic_messages_before(self, keep_ids: List[int]) -> int:
        """Mark older pydantic_ai_msg rows as archived so they aren't reloaded.

        We don't actually delete (keep audit trail); we null out the BLOB so
        load_pydantic_messages skips them. Returns rows touched.
        """
        if not keep_ids:
            return 0
        with self.get_session() as session:
            count = (
                session.query(Message)
                .filter(Message.pydantic_ai_msg.isnot(None))
                .filter(~Message.id.in_(keep_ids))
                .update({Message.pydantic_ai_msg: None}, synchronize_session=False)
            )
            session.commit()
            return count

    def archive_all_pydantic_messages(
        self, session_key: Optional[str] = None, include_legacy: bool = True
    ) -> int:
        """Null out the BLOB for every active pydantic_ai_msg row.

        Used by /compact when we replace the entire conversation snapshot with
        [summary, *tail]. Returns the number of rows archived.
        """
        with self.get_session() as session:
            query = session.query(Message).filter(Message.pydantic_ai_msg.isnot(None))
            if session_key is not None:
                if include_legacy:
                    query = query.filter(
                        (Message.session_key == session_key) | (Message.session_key.is_(None))
                    )
                else:
                    query = query.filter(Message.session_key == session_key)
            count = query.update({Message.pydantic_ai_msg: None}, synchronize_session=False)
            session.commit()
            return count

    def get_pydantic_message_ids_after(self, after_id: int) -> List[int]:
        """Return ids of rows with pydantic_ai_msg.id > after_id (chronological tail)."""
        with self.get_session() as session:
            rows = (
                session.query(Message.id)
                .filter(Message.pydantic_ai_msg.isnot(None))
                .filter(Message.id > after_id)
                .order_by(Message.id.asc())
                .all()
            )
        return [r.id for r in rows]

    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics."""
        with self.get_session() as session:
            total = session.query(Message).count()
            return {"total_messages": total}

    def archive_messages_before(self, before: datetime) -> int:
        """Hide messages before cutoff without deleting their audit rows."""
        with self.get_session() as session:
            affected = (
                session.query(Message)
                .filter(Message.context_visible == 1)
                .filter(Message.created_at < before)
                .update({Message.context_visible: 0}, synchronize_session=False)
            )
            session.commit()
            return affected

    def get_maintenance_backlog(
        self,
        message_before: datetime,
        *,
        event_before: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Return visible message and non-open event counts before cutoffs."""
        event_before = event_before or message_before
        with self.get_session() as session:
            old_messages = (
                session.query(Message)
                .filter(Message.context_visible == 1)
                .filter(Message.created_at < message_before)
                .count()
            )
            old_archivable_events = (
                session.query(Event)
                .filter(Event.context_visible == 1)
                .filter(Event.status.in_(["resolved", "promoted"]))
                .filter(Event.created_at < event_before)
                .count()
            )
            return {
                "old_messages": old_messages,
                "old_archivable_events": old_archivable_events,
            }

    def archive_closed_events_before(self, before: datetime) -> int:
        """Hide resolved/promoted events before cutoff without deleting history."""
        with self.get_session() as session:
            affected = (
                session.query(Event)
                .filter(Event.context_visible == 1)
                .filter(Event.status.in_(["resolved", "promoted"]))
                .filter(Event.created_at < before)
                .update({Event.context_visible: 0}, synchronize_session=False)
            )
            session.commit()
            return affected

    def delete_messages_before(self, before: datetime) -> int:
        """Delete messages before a given time."""
        with self.get_session() as session:
            count = session.query(Message).filter(
                Message.created_at < before
            ).delete()
            session.commit()
            return count

    # Scheduled task operations
    def create_scheduled_task(
        self,
        task_id: str,
        cron: str,
        prompt: str,
        protected: bool = False,
        *,
        handler: str = "agent",
        builtin_task: Optional[str] = None,
    ) -> ScheduledTask:
        """Create a scheduled task."""
        with self.get_session() as session:
            task = ScheduledTask(
                id=task_id,
                cron=cron,
                prompt=prompt,
                protected=1 if protected else 0,
                handler=handler,
                builtin_task=builtin_task,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

    def get_scheduled_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get one scheduled task by id."""
        with self.get_session() as session:
            return session.get(ScheduledTask, task_id)

    def get_scheduled_tasks(self, enabled_only: bool = True) -> List[ScheduledTask]:
        """Get scheduled tasks."""
        with self.get_session() as session:
            query = session.query(ScheduledTask)
            if enabled_only:
                query = query.filter(ScheduledTask.enabled == 1)
            return query.all()

    def update_scheduled_task(self, task_id: str, **kwargs) -> Optional[ScheduledTask]:
        """Update a scheduled task."""
        with self.get_session() as session:
            task = session.query(ScheduledTask).filter(
                ScheduledTask.id == task_id
            ).first()
            if task:
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                session.commit()
                session.refresh(task)
            return task

    def delete_scheduled_task(self, task_id: str, force: bool = False) -> bool:
        """Delete a scheduled task.

        By default, protected rows (provenance: config.yaml) are refused —
        this guards the schedule_task LLM tool from wiping built-ins. The
        scheduler's startup orphan-sweep is the one legitimate caller that
        needs to bypass this check (yaml deleted the row, so the protection
        no longer applies), and passes force=True.
        """
        with self.get_session() as session:
            task = session.query(ScheduledTask).filter(
                ScheduledTask.id == task_id
            ).first()
            if task and (force or not task.protected):
                session.delete(task)
                session.commit()
                return True
            return False

    # Generic subagent run operations
    def create_subagent_run(
        self,
        run_id: str,
        agent_name: str,
        agent_kind: str,
        engine: str,
        input_payload: Dict[str, Any],
        subject: Optional[str] = None,
        origin_channel: str = "cli",
        user_id: Optional[str] = None,
    ) -> SubAgentRun:
        """Create a persisted background subagent run row."""
        with self.get_session() as session:
            run = SubAgentRun(
                id=run_id,
                agent_name=agent_name,
                agent_kind=agent_kind,
                engine=engine,
                input_json=json.dumps(input_payload, ensure_ascii=False, default=str),
                subject=subject,
                status="pending",
                origin_channel=origin_channel,
                user_id=user_id,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def update_subagent_run(self, run_id: str, **kwargs) -> Optional[SubAgentRun]:
        """Update a subagent run and bump updated_at."""
        with self.get_session() as session:
            run = session.query(SubAgentRun).filter(SubAgentRun.id == run_id).first()
            if not run:
                return None
            for key, value in kwargs.items():
                if key == "input_payload":
                    run.input_json = json.dumps(
                        value or {}, ensure_ascii=False, default=str
                    )
                elif hasattr(run, key):
                    setattr(run, key, value)
            run.updated_at = _utcnow()
            if kwargs.get("status") in ("succeeded", "failed", "cancelled"):
                run.completed_at = _utcnow()
            session.commit()
            session.refresh(run)
            return run

    def get_subagent_run(self, run_id: str) -> Optional[SubAgentRun]:
        """Return one subagent run."""
        with self.get_session() as session:
            return session.query(SubAgentRun).filter(SubAgentRun.id == run_id).first()

    def list_subagent_runs(
        self, agent_name: Optional[str] = None, limit: int = 10
    ) -> List[SubAgentRun]:
        """Return recent subagent runs, optionally filtered by agent_name."""
        with self.get_session() as session:
            query = session.query(SubAgentRun)
            if agent_name:
                query = query.filter(SubAgentRun.agent_name == agent_name)
            return query.order_by(SubAgentRun.created_at.desc()).limit(limit).all()

    def list_subagent_runs_by_status(
        self,
        statuses: List[str],
        agent_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[SubAgentRun]:
        """Return subagent runs matching any status, oldest first."""
        with self.get_session() as session:
            query = session.query(SubAgentRun).filter(SubAgentRun.status.in_(statuses))
            if agent_name:
                query = query.filter(SubAgentRun.agent_name == agent_name)
            return query.order_by(SubAgentRun.created_at.asc()).limit(limit).all()

    # Agent event operations
    def create_agent_event(
        self,
        event_type: str,
        origin: str,
        run_id: Optional[str] = None,
        subject: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """Append one local observability event."""
        payload_json = None
        if payload is not None:
            payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        with self.get_session() as session:
            event = AgentEvent(
                run_id=run_id,
                origin=origin,
                type=event_type,
                subject=subject,
                payload_json=payload_json,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def get_agent_events(self, limit: int = 100) -> List[AgentEvent]:
        """Return recent local observability events."""
        with self.get_session() as session:
            return (
                session.query(AgentEvent)
                .order_by(AgentEvent.created_at.desc(), AgentEvent.id.desc())
                .limit(limit)
                .all()
            )

    def get_agent_events_since(
        self,
        since: datetime,
        *,
        event_type: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> List[AgentEvent]:
        """Return observability events since a UTC-naive timestamp."""
        with self.get_session() as session:
            query = session.query(AgentEvent).filter(AgentEvent.created_at >= since)
            if event_type is not None:
                query = query.filter(AgentEvent.type == event_type)
            if origin is not None:
                query = query.filter(AgentEvent.origin == origin)
            return query.order_by(
                AgentEvent.created_at.desc(), AgentEvent.id.desc()
            ).all()

    def execute_query(self, sql: str, params: Optional[List] = None) -> List[Dict]:
        """Execute a read-only query."""
        from sqlalchemy import text
        with self.get_session() as session:
            # Convert list params to dict format for SQLAlchemy 2.0
            if params and isinstance(params, list):
                named_params = {str(i): v for i, v in enumerate(params)}
                for i in range(len(params)):
                    sql = sql.replace("?", f":{i}", 1)
                result = session.execute(text(sql), named_params)
            else:
                result = session.execute(text(sql), params or {})
            if result.returns_rows:
                return [dict(row._mapping) for row in result]
            return []

    def execute_statement(self, sql: str, params: Optional[List] = None) -> int:
        """Execute a write statement."""
        from sqlalchemy import text
        with self.get_session() as session:
            # Convert list params to dict format for SQLAlchemy 2.0
            if params and isinstance(params, list):
                # Convert positional params to named params
                named_params = {str(i): v for i, v in enumerate(params)}
                # Replace ? with :0, :1, etc.
                for i in range(len(params)):
                    sql = sql.replace("?", f":{i}", 1)
                result = session.execute(text(sql), named_params)
            else:
                result = session.execute(text(sql), params or {})
            session.commit()
            return result.rowcount


def _preview_for_message(msg: ModelMessage) -> str:
    """Best-effort short text preview for a ModelMessage (for human inspection)."""
    # Compacted snapshots store the summary as a SystemPromptPart after the
    # leading system prompts. Prefer it in the human preview so DB viewers make
    # the active summary row obvious.
    for part in getattr(msg, "parts", []):
        text = getattr(part, "content", None)
        if isinstance(text, str) and "Summary of previous conversation:" in text:
            return text[:500]

    pieces = []
    for part in getattr(msg, "parts", []):
        text = getattr(part, "content", None)
        if isinstance(text, str):
            pieces.append(text)
        elif text is not None:
            pieces.append(repr(text)[:200])
    return " | ".join(pieces)[:500]


# Database singleton (so module-level callers can reach the active db without
# re-opening the file)
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Return the process-wide Database instance, creating it on first call."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def set_db(db: Database) -> None:
    """Override the singleton (used by tests)."""
    global _db_instance
    _db_instance = db
