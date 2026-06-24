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

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, BLOB, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

import tiktoken
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ThinkingPart,
)

from config import get_config

logger = logging.getLogger(__name__)

Base = declarative_base()

_token_encoding = None


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


def _response_has_actionable_part(msg: ModelResponse) -> bool:
    """True when a response can be replayed as assistant content/tool_calls."""
    for part in msg.parts:
        kind = getattr(part, "part_kind", "")
        if kind == "text" and bool(getattr(part, "content", None)):
            return True
        if kind == "tool-call":
            return True
    return False


def _strip_thinking_from_response(msg: ModelResponse) -> Optional[ModelResponse]:
    """Remove thinking parts and drop responses that would be invalid history."""
    parts = [part for part in msg.parts if not isinstance(part, ThinkingPart)]
    if not parts:
        return None
    cleaned = msg if len(parts) == len(msg.parts) else replace(msg, parts=parts)
    if not _response_has_actionable_part(cleaned):
        return None
    return cleaned


def sanitize_pydantic_messages_for_history(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """Return only model messages safe to persist and replay as history.

    DeepSeek/OpenAI-compatible chat history cannot contain an assistant message
    that has reasoning/thinking but no visible content or tool_calls. We also
    avoid persisting thinking parts in otherwise valid responses so historical
    replay stays provider-portable and cannot regress into content=None turns.
    """
    sanitized: List[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            cleaned = _strip_thinking_from_response(msg)
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


class Event(Base):
    """事件记录表 - Compatible with original schema."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)  # remind | check | response | note | triggered
    status = Column(String, nullable=False, default="logged", server_default="logged")
    content = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


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


class ScheduledTask(Base):
    """定时任务表 - Compatible with original schema."""
    __tablename__ = "scheduled_tasks"

    id = Column(String, primary_key=True)
    cron = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    enabled = Column(Integer, default=1)
    protected = Column(Integer, default=0)
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

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    # Event operations
    def create_event(
        self, event_type: str, content: str, status: str = "logged"
    ) -> Event:
        """Create a new event."""
        with self.get_session() as session:
            event = Event(
                type=event_type,
                status=status,
                content=content,
            )
            session.add(event)
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
        """Get recent events while omitting closed/noisy status buckets."""
        with self.get_session() as session:
            query = session.query(Event)
            if excluded_statuses:
                query = query.filter(Event.status.notin_(excluded_statuses))
            return query.order_by(Event.created_at.desc()).limit(limit).all()

    def get_events_by_status(self, status: str, limit: int = 200) -> List[Event]:
        """Get events by attention status, newest first."""
        with self.get_session() as session:
            return (
                session.query(Event)
                .filter(Event.status == status)
                .order_by(Event.created_at.desc())
                .limit(limit)
                .all()
            )

    def count_events_by_status(self, status: str) -> int:
        """Count events by attention status."""
        with self.get_session() as session:
            return session.query(Event).filter(Event.status == status).count()

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
    ) -> None:
        """Persist a batch of Pydantic AI messages from one agent run.

        Each ModelMessage is stored as one row, in order, so load_pydantic_messages
        can reconstruct the conversation. The pydantic_ai_msg BLOB column holds
        the canonical serialized form; content holds a short text preview for humans.
        """
        messages = sanitize_pydantic_messages_for_history(messages)
        if not messages:
            return
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
                    )
                )
            session.commit()

    def load_pydantic_messages(
        self, token_budget: Optional[int] = None, batch_size: int = 100
    ) -> List[ModelMessage]:
        """Load recent pydantic-ai messages newest-first up to a token budget.

        There is no fixed message-count cap: depth emerges from the token
        budget (defaults to history.context_tokens). Rows are walked
        newest-first; once the budget is exceeded we stop, so the oldest rows
        beyond the budget are simply not loaded. The newest row is always
        included even if it alone exceeds the budget.

        Pre-run compaction is the real safety net — this budget only bounds how
        much we deserialize and feed it. Rows without a pydantic_ai_msg payload
        (legacy plain-text rows, or rows archived by /compact) are skipped.
        """
        if token_budget is None:
            token_budget = get_config().history.context_tokens

        selected: List[List[ModelMessage]] = []
        total = 0
        offset = 0

        with self.get_session() as session:
            while True:
                rows = (
                    session.query(Message.id, Message.pydantic_ai_msg)
                    .filter(Message.pydantic_ai_msg.isnot(None))
                    .order_by(Message.id.desc())
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
                    msgs = sanitize_pydantic_messages_for_history(msgs)
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
                    selected.append(msgs)
                    total += cost

                if should_stop or len(rows) < batch_size:
                    break
                offset += batch_size

        selected.reverse()  # back to chronological order
        result: List[ModelMessage] = []
        for msgs in selected:
            result.extend(msgs)
        return result

    def archive_invalid_pydantic_messages(self) -> int:
        """Sanitize active history blobs and archive rows that cannot be replayed.

        Rows that become empty after sanitization have their BLOB nulled so
        load_pydantic_messages skips them while the human-readable audit preview
        remains in `content`. Rows that only needed thinking parts removed are
        rewritten with the sanitized blob.
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

                sanitized = sanitize_pydantic_messages_for_history(msgs)
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

    def archive_all_pydantic_messages(self) -> int:
        """Null out the BLOB for every active pydantic_ai_msg row.

        Used by /compact when we replace the entire conversation snapshot with
        [summary, *tail]. Returns the number of rows archived.
        """
        with self.get_session() as session:
            count = (
                session.query(Message)
                .filter(Message.pydantic_ai_msg.isnot(None))
                .update({Message.pydantic_ai_msg: None}, synchronize_session=False)
            )
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

    def delete_messages_before(self, before: datetime) -> int:
        """Delete messages before a given time."""
        with self.get_session() as session:
            count = session.query(Message).filter(
                Message.created_at < before
            ).delete()
            session.commit()
            return count

    # Scheduled task operations
    def create_scheduled_task(self, task_id: str, cron: str,
                              prompt: str, protected: bool = False) -> ScheduledTask:
        """Create a scheduled task."""
        with self.get_session() as session:
            task = ScheduledTask(
                id=task_id,
                cron=cron,
                prompt=prompt,
                protected=1 if protected else 0,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

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
