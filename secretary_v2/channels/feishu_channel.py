"""Feishu/Lark channel for secretary v2."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Optional, Tuple
from urllib.parse import quote

from .base import Channel, IncomingMessage, is_no_action_response
from i18n import resolve_ui_language, t

logger = logging.getLogger(__name__)


def _ensure_lark_ws_loop_not_running() -> None:
    """Avoid lark-oapi binding its module-level WS loop to our app loop.

    lark_oapi.ws.client stores an asyncio loop in a module global at import
    time, then later calls run_until_complete() on it from its blocking
    WebSocket start path. If the module was imported while our application
    event loop was already running, the SDK tries to run that same loop again
    and fails with "This event loop is already running".
    """
    try:
        import lark_oapi.ws.client as ws_client
    except Exception:
        return

    ws_loop = getattr(ws_client, "loop", None)
    if ws_loop is None or ws_loop.is_running() or ws_loop.is_closed():
        ws_client.loop = asyncio.new_event_loop()


def _plain_text_for_feishu(text: str) -> str:
    """Lightly remove common LLM Markdown decoration for plain Feishu sends."""
    if not text:
        return text
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    cleaned = re.sub(r"\*\*([^\n*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^\n_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^\n*]+)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`\n]+)`", r"\1", cleaned)
    return cleaned


class FeishuChannel(Channel):
    """Feishu/Lark channel implementation based on lark-oapi."""

    name = "feishu"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        default_chat_id: str,
        message_handler: Callable[[IncomingMessage], Awaitable[str]],
        domain: str = "https://open.feishu.cn",
        transport: str = "ws",
        encrypt_key: str = "",
        verification_token: str = "",
        require_mention: bool = True,
        allow_chat_ids: Optional[list[str]] = None,
        allow_sender_ids: Optional[list[str]] = None,
        peer_channel_names: Optional[list[str]] = None,
        outbox_capacity: int = 100,
        sdk_channel_factory: Optional[Callable[..., Any]] = None,
        http_client_factory: Optional[Callable[[], Any]] = None,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.default_chat_id = default_chat_id
        self.message_handler = message_handler
        self.domain = domain
        self.transport = transport or "ws"
        self.encrypt_key = encrypt_key or ""
        self.verification_token = verification_token or ""
        self.require_mention = require_mention
        self.allow_chat_ids = set(allow_chat_ids or [])
        self.allow_sender_ids = set(allow_sender_ids or [])
        self.peer_channel_names = list(peer_channel_names or ["feishu"])
        self.peer_channel_health_provider = None
        self._sdk_channel_factory = sdk_channel_factory
        self._http_client_factory = http_client_factory
        self._sdk_channel: Any = None
        self._running = False
        self._ready = False
        self._outbox: Deque[Tuple[str, str]] = deque(maxlen=outbox_capacity)
        self._file_outbox: Deque[Tuple[str, str, Optional[str]]] = deque(maxlen=outbox_capacity)
        self._drain_lock = asyncio.Lock()
        self._tenant_access_token: Optional[str] = None
        self._tenant_access_token_expires_at = 0.0

    async def start(self) -> None:
        """Start the Feishu channel.

        The lark-oapi channel owns WebSocket reconnects. This wrapper keeps
        parity with Telegram by marking readiness, draining outbound messages
        queued during startup, and keeping the channel task alive until stop().
        """
        self._running = True
        backoff = 5.0
        max_backoff = 120.0

        while self._running:
            try:
                self._sdk_channel = self._build_sdk_channel()
                self._register_handlers()
                await self._connect_until_ready()
                self._ready = True
                logger.info("Feishu channel started")
                drained = await self._drain_outbox()
                if drained:
                    logger.info(f"Drained {drained} buffered Feishu message(s)")
                backoff = 5.0

                while self._running:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._ready = False
                logger.error(f"Feishu channel failed: {e}")
                if not self._running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def stop(self) -> None:
        """Stop the Feishu channel."""
        self._running = False
        self._ready = False
        if self._sdk_channel is None:
            return
        disconnect = getattr(self._sdk_channel, "disconnect", None)
        stop = getattr(self._sdk_channel, "stop", None)
        try:
            if disconnect:
                result = disconnect()
                if inspect.isawaitable(result):
                    await result
            elif stop:
                result = stop()
                if inspect.isawaitable(result):
                    await result
        except Exception as e:
            logger.warning(f"Feishu channel stop failed: {e}")
        finally:
            logger.info("Feishu channel stopped")

    async def send(self, text: str, user_id: Optional[str] = None) -> None:
        """Send a plain-text message to Feishu, buffering while disconnected."""
        target_chat_id = user_id or self.default_chat_id
        chunks = self._split_message(text, 4000)

        if not target_chat_id:
            logger.warning("Feishu send skipped; no target chat id configured")
            return

        if not self._is_ready():
            for chunk in chunks:
                if len(self._outbox) == self._outbox.maxlen:
                    logger.warning("Feishu outbox full; dropping oldest queued message")
                self._outbox.append((chunk, target_chat_id))
            logger.info(
                f"Feishu not ready; buffered {len(chunks)} chunk(s) "
                f"(outbox={len(self._outbox)})"
            )
            return

        await self._send_chunks_now(chunks, target_chat_id)

    async def send_file(
        self,
        path: str | Path,
        caption: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Send a file to Feishu, buffering during startup/reconnect windows."""
        target_chat_id = user_id or self.default_chat_id
        file_path = Path(path)
        if not file_path.is_file():
            logger.warning(f"Feishu file send skipped; file missing: {file_path}")
            fallback = f"{caption or 'File send failed'}\n\nFile does not exist: `{file_path}`"
            await self.send(fallback, target_chat_id)
            return
        if not target_chat_id:
            logger.warning("Feishu file send skipped; no target chat id configured")
            return

        if not self._is_ready():
            if len(self._file_outbox) == self._file_outbox.maxlen:
                logger.warning("Feishu file outbox full; dropping oldest queued file")
            self._file_outbox.append((str(file_path), target_chat_id, caption))
            logger.info(
                f"Feishu not ready; buffered file '{file_path.name}' "
                f"(file_outbox={len(self._file_outbox)})"
            )
            return

        await self._send_file_now(file_path, target_chat_id, caption)

    def _build_sdk_channel(self):
        if self._sdk_channel_factory:
            return self._sdk_channel_factory(self)
        if self.transport != "ws":
            raise RuntimeError(
                "FeishuChannel currently supports transport='ws'. "
                "lark-oapi webhook transport needs an HTTP server route wired to "
                "handle_webhook_request()."
            )
        try:
            from lark_oapi.channel import FeishuChannel as LarkFeishuChannel
        except ImportError as e:
            raise RuntimeError(
                "lark-oapi is required for FeishuChannel. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from e
        _ensure_lark_ws_loop_not_running()

        kwargs = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
            "transport": self.transport,
        }
        if self.domain:
            kwargs["domain"] = self.domain
        if self.encrypt_key:
            kwargs["encrypt_key"] = self.encrypt_key
        if self.verification_token:
            kwargs["verification_token"] = self.verification_token
        return LarkFeishuChannel(**kwargs)

    def _register_handlers(self) -> None:
        self._sdk_channel.on("message", self._handle_message)
        self._sdk_channel.on("error", self._handle_error)

    async def _connect_until_ready(self) -> None:
        connect_until_ready = getattr(self._sdk_channel, "connect_until_ready", None)
        if connect_until_ready:
            result = connect_until_ready(timeout=30)
            if inspect.isawaitable(result):
                await result
            return
        connect = getattr(self._sdk_channel, "connect")
        result = connect()
        if inspect.isawaitable(result):
            await result

    def _is_ready(self) -> bool:
        return bool(self._running and self._ready and self._sdk_channel is not None)

    def health_status(self) -> str:
        if self._is_ready():
            return "healthy"
        return "starting" if self._running else "stopped"

    async def _send_chunks_now(self, chunks: list[str], target_chat_id: str) -> None:
        for chunk in chunks:
            plain_chunk = _plain_text_for_feishu(chunk)
            try:
                result = await self._sdk_channel.send(
                    target_chat_id,
                    {"text": plain_chunk},
                    {"receive_id_type": "chat_id"},
                )
                self._raise_if_send_failed(result)
            except Exception as err:
                logger.error(f"Feishu text send failed: {err}")
                if len(self._outbox) < self._outbox.maxlen:
                    self._outbox.append((chunk, target_chat_id))

    async def _send_file_now(
        self,
        path: Path,
        target_chat_id: str,
        caption: Optional[str] = None,
    ) -> None:
        try:
            if caption:
                await self._send_chunks_now([caption], target_chat_id)
            result = await self._sdk_channel.send(
                target_chat_id,
                {"file": {"source": str(path), "file_name": path.name}},
                {"receive_id_type": "chat_id"},
            )
            self._raise_if_send_failed(result)
        except Exception as err:
            logger.error(f"Feishu file send failed for {path}: {err}")
            fallback = f"{caption or 'File send failed'}\n\nReport file: `{path}`"
            await self.send(fallback, target_chat_id)

    async def _add_message_reaction(
        self,
        message_id: Optional[str],
        emoji_type: str = "THUMBSUP",
    ) -> bool:
        if not message_id:
            return False
        try:
            await self._feishu_api_post(
                f"/open-apis/im/v1/messages/{quote(str(message_id), safe='')}/reactions",
                {"reaction_type": {"emoji_type": emoji_type}},
            )
            return True
        except Exception as err:
            logger.warning(f"Feishu reaction skipped for {message_id}: {err}")
            return False

    async def _reply_to_message(self, text: str, message_id: Optional[str]) -> bool:
        if not message_id:
            return False

        chunks = self._split_message(text, 4000)
        try:
            for chunk in chunks:
                plain_chunk = _plain_text_for_feishu(chunk)
                content = json.dumps({"text": plain_chunk}, ensure_ascii=False)
                await self._feishu_api_post(
                    f"/open-apis/im/v1/messages/{quote(str(message_id), safe='')}/reply",
                    {
                        "msg_type": "text",
                        "content": content,
                        "uuid": str(uuid.uuid4()),
                    },
                )
            return True
        except Exception as err:
            logger.warning(
                f"Feishu reply failed for {message_id}; falling back to send: {err}"
            )
            return False

    async def _feishu_api_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = await self._get_tenant_access_token()
        return await self._post_json(
            path,
            payload,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def _get_tenant_access_token(self) -> str:
        now = time.monotonic()
        if self._tenant_access_token and now < self._tenant_access_token_expires_at:
            return self._tenant_access_token

        data = await self._post_json(
            "/open-apis/auth/v3/tenant_access_token/internal",
            {"app_id": self.app_id, "app_secret": self.app_secret},
            headers={},
        )
        token = data.get("tenant_access_token")
        if not token:
            raise RuntimeError("Feishu tenant_access_token missing from response")
        try:
            expires_in = int(data.get("expire", 7200))
        except (TypeError, ValueError):
            expires_in = 7200
        self._tenant_access_token = str(token)
        self._tenant_access_token_expires_at = now + max(expires_in - 300, 60)
        return self._tenant_access_token

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if self._http_client_factory is None and not self._has_live_lark_sdk_channel():
            raise RuntimeError("Feishu REST API requires a live lark-oapi channel")

        client = None
        close_client = False
        try:
            if self._http_client_factory is not None:
                client = self._http_client_factory()
            else:
                import httpx

                client = httpx.AsyncClient(base_url=self.domain, timeout=10)
                close_client = True

            request_headers = {"Content-Type": "application/json; charset=utf-8"}
            request_headers.update(headers)
            response = await client.post(path, json=payload, headers=request_headers)
            if isinstance(response, dict):
                data = response
            else:
                response.raise_for_status()
                data = response.json()
            code = data.get("code", 0)
            if code != 0:
                raise RuntimeError(data.get("msg") or data)
            return data
        finally:
            if close_client and client is not None:
                await client.aclose()

    def _has_live_lark_sdk_channel(self) -> bool:
        if self._sdk_channel is None:
            return False
        module = self._sdk_channel.__class__.__module__
        return module.startswith("lark_oapi")

    async def _drain_outbox(self) -> int:
        async with self._drain_lock:
            if (not self._outbox and not self._file_outbox) or not self._is_ready():
                return 0
            sent = 0
            pending = list(self._outbox)
            self._outbox.clear()
            for chunk, chat_id in pending:
                await self._send_chunks_now([chunk], chat_id)
                sent += 1
            pending_files = list(self._file_outbox)
            self._file_outbox.clear()
            for path, chat_id, caption in pending_files:
                await self._send_file_now(Path(path), chat_id, caption)
                sent += 1
            return sent

    def _split_message(self, text: str, max_length: int) -> list[str]:
        if len(text) <= max_length:
            return [text]

        messages = []
        while text:
            if len(text) <= max_length:
                messages.append(text)
                break
            split_at = text.rfind("\n", 0, max_length)
            if split_at == -1:
                split_at = max_length
            messages.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return messages

    async def _handle_message(self, msg: Any) -> None:
        text = (getattr(msg, "content_text", None) or "").strip()
        if not text:
            return

        chat_id = self._message_chat_id(msg)
        sender_id = self._message_sender_id(msg)
        if not self._message_allowed(msg, chat_id, sender_id):
            return

        command = self._parse_command(text)
        if command:
            await self._handle_command(command, msg, chat_id, sender_id)
            return

        source_message_id = getattr(msg, "message_id", None)
        await self._add_message_reaction(source_message_id, "THUMBSUP")

        message = IncomingMessage(
            text=text,
            channel="feishu",
            user_id=sender_id,
            conversation_id=chat_id,
            reply_to_id=source_message_id,
            thread_id=getattr(getattr(msg, "conversation", None), "thread_id", None),
            metadata={
                "chat_id": chat_id,
                "message_id": source_message_id,
                "sender_id": sender_id,
                "sender_name": getattr(msg, "sender_name", None),
                "chat_type": getattr(msg, "chat_type", None),
            },
        )

        try:
            response = await self.message_handler(message)
            if is_no_action_response(response):
                logger.warning(
                    "Suppressing internal NO_ACTION final response for Feishu user message"
                )
                return
            if await self._reply_to_message(response, source_message_id):
                return
            await self.send(response, chat_id)
        except Exception as e:
            logger.error(f"Error handling Feishu message: {e}")
            error_text = t("telegram.message.error", self._ui_lang())
            if not await self._reply_to_message(error_text, source_message_id):
                await self.send(error_text, chat_id)

    async def _handle_command(
        self, command: str, msg: Any, chat_id: str, sender_id: str
    ) -> None:
        from channel_commands import CommandScope

        scope = CommandScope(
            channel="feishu",
            user_id=sender_id,
            conversation_id=chat_id,
            thread_id=getattr(getattr(msg, "conversation", None), "thread_id", None),
        )
        if command == "start":
            await self.send(t("telegram.start", self._ui_lang()), chat_id)
            return
        if command == "help":
            await self.send(t("telegram.help", self._ui_lang()), chat_id)
            return
        if command == "status":
            await self._status_command(chat_id, scope)
            return
        if command == "skills":
            await self._skills_command(chat_id)
            return
        if command == "compact":
            await self._compact_command(chat_id, scope)

    async def _status_command(self, chat_id: str, scope) -> None:
        from channel_commands import build_status_text

        lang = self._ui_lang()
        try:
            status_text = build_status_text(
                scope=scope,
                lang=lang,
                peer_channel_names=self.peer_channel_names,
                channel_health=(
                    self.peer_channel_health_provider()
                    if self.peer_channel_health_provider
                    else None
                ),
            )
        except Exception as e:
            logger.error(f"Feishu /status failed: {e}")
            await self.send(t("command.status.read_failed", lang), chat_id)
            return
        await self.send(status_text, chat_id)

    async def _skills_command(self, chat_id: str) -> None:
        from skills_loader import get_skills_loader

        lang = self._ui_lang()
        skills = get_skills_loader().get_all_skills()
        if not skills:
            await self.send(t("telegram.skills.empty", lang), chat_id)
            return

        lines = [t("telegram.skills.title", lang, count=len(skills))]
        for name, meta in skills.items():
            triggers = ", ".join(meta.triggers[:3]) if meta.triggers else t("telegram.skills.no_triggers", lang)
            lines.append(f"- {name} - {triggers}")
        await self.send("\n".join(lines), chat_id)

    async def _compact_command(self, chat_id: str, scope) -> None:
        from channel_commands import compact_conversation

        lang = self._ui_lang()
        await self.send(t("command.compact.running", lang), chat_id)
        try:
            result = await compact_conversation(scope=scope, lang=lang)
        except Exception as e:
            logger.error(f"Feishu /compact failed: {e}")
            await self.send(t("command.compact.failed", lang, error=e), chat_id)
            return
        await self.send(result, chat_id)

    def _message_allowed(self, msg: Any, chat_id: str, sender_id: str) -> bool:
        if self.allow_chat_ids and chat_id not in self.allow_chat_ids:
            logger.info(f"Feishu message ignored from unallowlisted chat {chat_id}")
            return False
        if self.allow_sender_ids and sender_id not in self.allow_sender_ids:
            logger.info(f"Feishu message ignored from unallowlisted sender {sender_id}")
            return False
        chat_type = (getattr(msg, "chat_type", "") or "").lower()
        is_group = chat_type in {"group", "topic"}
        if is_group and self.require_mention and not getattr(msg, "mentioned_bot", False):
            return False
        return True

    def _message_chat_id(self, msg: Any) -> str:
        chat_id = getattr(msg, "chat_id", None)
        if chat_id:
            return str(chat_id)
        conversation = getattr(msg, "conversation", None)
        return str(getattr(conversation, "chat_id", "") or "")

    def _message_sender_id(self, msg: Any) -> str:
        sender_id = getattr(msg, "sender_id", None)
        if sender_id:
            return str(sender_id)
        sender = getattr(msg, "sender", None)
        return str(getattr(sender, "open_id", "") or getattr(sender, "user_id", "") or "")

    def _parse_command(self, text: str) -> Optional[str]:
        first = text.strip().split(maxsplit=1)[0].lower()
        if not first.startswith("/"):
            return None
        command = first[1:].split("@", 1)[0]
        if command in {"start", "help", "status", "skills", "compact"}:
            return command
        return None

    def _ui_lang(self) -> str:
        from config import get_config

        cfg = get_config()
        return resolve_ui_language(cfg.ui_language, agent_language=cfg.language)

    async def _handle_error(self, err: Any) -> None:
        logger.error(f"Feishu SDK error: {err}")

    @staticmethod
    def _raise_if_send_failed(result: Any) -> None:
        if result is None:
            return
        success = getattr(result, "success", None)
        if success is False:
            error = getattr(result, "error", None)
            raise RuntimeError(error or "Feishu send failed")
