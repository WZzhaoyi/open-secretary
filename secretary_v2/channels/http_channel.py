"""HTTP channel for secretary v2.

HTTP is fundamentally request/response, so 'send' is best-effort: outbound
messages get queued in memory and clients drain them via GET /messages.
That makes the channel symmetric enough that scheduler-triggered messages
can still be observed when only the HTTP channel is configured.

Bind defaults to 127.0.0.1 (localhost only); changing this is opt-in via
the `bind_host` argument so a stray deploy doesn't accidentally expose the
webhook to the network.
"""

import asyncio
import logging
import uuid
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Optional, Union

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .base import Channel, IncomingMessage, is_no_action_response

logger = logging.getLogger(__name__)

ResponseChannel = Union[Channel, Callable[[], Optional[Channel]]]
EventRecorder = Callable[..., Any]
_WEBHOOK_RECORD_STATUSES = {"logged", "open"}


class WebhookMessage(BaseModel):
    message: str
    summary: Optional[str] = None
    record: Optional[Union[str, bool]] = None
    user_id: str = "webhook_user"


def _session_part(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text.replace(":", "_")


def _webhook_session_key(user_id: str, agent_id: str = "secretary") -> str:
    return f"agent:{_session_part(agent_id)}:http:conversation:{_session_part(user_id)}"


def _normalize_record_status(record: object) -> Optional[str]:
    """Return the events.status requested by a webhook record value."""
    if record is None or record is False:
        return None
    if isinstance(record, str):
        normalized = record.strip().lower()
        if normalized in {"", "false", "none", "no", "off"}:
            return None
        if normalized in _WEBHOOK_RECORD_STATUSES:
            return normalized
    raise ValueError("record must be omitted, false, 'logged', or 'open'")


class HTTPChannel(Channel):
    """HTTP webhook channel."""

    name = "http"

    def __init__(
        self,
        token: str,
        message_handler: Callable[[IncomingMessage], Awaitable[str]],
        response_channel: Optional[ResponseChannel] = None,
        event_recorder: Optional[EventRecorder] = None,
        port: int = 11269,
        bind_host: str = "127.0.0.1",
        outbox_capacity: int = 100,
    ):
        self.token = token
        self.message_handler = message_handler
        self.response_channel = response_channel
        self.event_recorder = event_recorder
        self.port = port
        self.bind_host = bind_host
        self.app = FastAPI()
        self._server: Optional[uvicorn.Server] = None
        self._outbox: Deque[dict] = deque(maxlen=outbox_capacity)
        self._webhook_tasks: set[asyncio.Task] = set()
        self._setup_routes()

    def _get_response_channel(self) -> Optional[Channel]:
        if callable(self.response_channel):
            return self.response_channel()
        return self.response_channel

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "outbox_pending": len(self._outbox)}

        @self.app.post("/hooks")
        async def webhook(
            message: WebhookMessage,
            x_webhook_token: Optional[str] = Header(None),
        ):
            if x_webhook_token != self.token:
                raise HTTPException(status_code=401, detail="Invalid token")
            try:
                record_status = _normalize_record_status(message.record)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

            run_id = f"hook_{uuid.uuid4().hex[:12]}"
            session_key = _webhook_session_key(message.user_id)
            metadata = {
                "webhook_run_id": run_id,
                "user_id": message.user_id,
            }
            if record_status:
                metadata["record"] = record_status
                event = self._record_webhook_event(
                    message=message,
                    record_status=record_status,
                    session_key=session_key,
                    metadata=metadata,
                )
                if event is not None:
                    event_id = getattr(event, "id", None)
                    if event_id is not None:
                        metadata["recorded_event_id"] = event_id
                    metadata["recorded_event_summary"] = getattr(event, "summary", None)
                    metadata["summary_supplied"] = bool(message.summary)
            incoming = IncomingMessage(
                text=message.message,
                channel="http",
                user_id=message.user_id,
                conversation_id=message.user_id,
                metadata=metadata,
            )
            task = asyncio.create_task(self._handle_webhook_async(run_id, incoming))
            self._webhook_tasks.add(task)
            task.add_done_callback(self._webhook_tasks.discard)
            return JSONResponse(
                status_code=202,
                content={"ok": True, "runId": run_id},
            )

        @self.app.get("/messages")
        async def drain_messages(
            x_webhook_token: Optional[str] = Header(None),
        ):
            """Pop and return any messages the agent pushed via send().

            Only intended for the local owner of the bot. Auth via the same token
            used for /hooks; we still gate it because the messages may contain
            private content.
            """
            if x_webhook_token != self.token:
                raise HTTPException(status_code=401, detail="Invalid token")
            drained = list(self._outbox)
            self._outbox.clear()
            return {"messages": drained}

    def _record_webhook_event(
        self,
        *,
        message: WebhookMessage,
        record_status: str,
        session_key: str,
        metadata: dict,
    ) -> Any:
        if self.event_recorder is None:
            logger.warning(
                "Webhook requested record=%s but no recorder is configured",
                record_status,
            )
            return None
        return self.event_recorder(
            "note",
            message.message,
            status=record_status,
            summary=message.summary,
            source_channel="http",
            session_key=session_key,
            metadata=metadata,
        )

    async def _handle_webhook_async(
        self, run_id: str, incoming: IncomingMessage
    ) -> None:
        try:
            response = await self.message_handler(incoming)
        except Exception as e:
            logger.error(f"Error handling webhook run {run_id}: {e}")
            response = f"Webhook handling failed ({run_id}): {e}"

        if is_no_action_response(response):
            logger.info(
                "Suppressing internal NO_ACTION final response for webhook run %s",
                run_id,
            )
            return

        response_channel = self._get_response_channel()
        if response_channel is None:
            logger.warning("No response channel configured for webhook run %s", run_id)
            return

        try:
            await response_channel.send(response, user_id=None)
        except Exception as e:
            logger.error(f"Error delivering webhook run {run_id}: {e}")

    async def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.bind_host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        logger.info(f"HTTP channel starting on {self.bind_host}:{self.port}")
        await self._server.serve()

    def health_status(self) -> str:
        if self._server is None:
            return "stopped"
        if getattr(self._server, "started", False) and not self._server.should_exit:
            return "healthy"
        return "starting" if not self._server.should_exit else "stopped"

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        logger.info("HTTP channel stopping")

    async def send(self, text: str, user_id: Optional[str] = None) -> None:
        """Buffer the message for the next /messages drain.

        If the outbox is full, the oldest message is dropped — caller has no way
        to be notified, so we log it.
        """
        if len(self._outbox) == self._outbox.maxlen:
            logger.warning("HTTP outbox full; dropping oldest message")
        self._outbox.append({"text": text, "user_id": user_id or "default"})
        logger.debug(f"HTTP outbox queued (now {len(self._outbox)})")
