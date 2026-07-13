"""Telegram channel for secretary v2."""

import asyncio
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable, Deque, Optional, Tuple

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

from .base import Channel, IncomingMessage, is_no_action_response
from i18n import resolve_ui_language, t

logger = logging.getLogger(__name__)


def _plain_text_for_telegram(text: str) -> str:
    """Remove common LLM Markdown decoration before plain Telegram sends."""
    if not text:
        return text
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    cleaned = re.sub(r"\*\*([^\n*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^\n_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^\n*]+)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`\n]+)`", r"\1", cleaned)
    return cleaned

# How long since the last successful getUpdates round-trip before we consider
# the polling task wedged. Long-poll defaults to 10s; a healthy poll completes
# at least every ~30s. 90s is comfortably past that without false positives.
GET_UPDATES_STALL_THRESHOLD_SEC = 90.0


class _StallTrackedRequest(HTTPXRequest):
    """HTTPXRequest that records a monotonic timestamp on every successful
    HTTP round-trip. Used as the `get_updates_request` for the Application so
    the watchdog can observe whether long-polling is still actually issuing
    network calls — the stock `Updater.running` flag stays True even when the
    underlying httpx socket has silently wedged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize to "now" so the first 90s after build doesn't trip the
        # watchdog before the first poll completes.
        self.last_seen_at: float = time.monotonic()

    async def do_request(self, *args, **kwargs):
        result = await super().do_request(*args, **kwargs)
        self.last_seen_at = time.monotonic()
        return result

def bot_commands(lang: str) -> list[BotCommand]:
    return [
        BotCommand("start", t("telegram.command.start", lang)),
        BotCommand("help", t("telegram.command.help", lang)),
        BotCommand("status", t("telegram.command.status", lang)),
        BotCommand("skills", t("telegram.command.skills", lang)),
        BotCommand("compact", t("telegram.command.compact", lang)),
    ]


class TelegramChannel(Channel):
    """Telegram channel implementation."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        message_handler: Callable[[IncomingMessage], Awaitable[str]],
        peer_channel_names: Optional[list] = None,
        outbox_capacity: int = 100,
        proxy: str = "",
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        # Outbound proxy URL for both the bot API pool and long-polling,
        # e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:1080.
        self.proxy = (proxy or "").strip()
        self.message_handler = message_handler
        # Names of all channels running alongside this one — surfaced via /status
        # so users can see the full I/O footprint (e.g. "telegram + http").
        self.peer_channel_names = list(peer_channel_names or ["telegram"])
        self.peer_channel_health_provider = None
        self.app: Optional[Application] = None
        self._running = False
        # Reference to the Application's getUpdates HTTPX request. Holds
        # `last_seen_at` for the polling-stall watchdog. Replaced on every
        # rebuild; cleared on teardown.
        self._get_updates_request: Optional[_StallTrackedRequest] = None
        # Outbox for messages produced while the Updater is being torn down /
        # rebuilt by the watchdog. Without this, the agent's send_message tool
        # silently drops messages during the 5–120s reconnect window.
        # Each text entry: (text, target_chat_id)
        self._outbox: Deque[Tuple[str, str]] = deque(maxlen=outbox_capacity)
        # Each file entry: (path, target_chat_id, caption)
        self._file_outbox: Deque[Tuple[str, str, Optional[str]]] = deque(maxlen=outbox_capacity)
        self._drain_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the Telegram bot.

        Includes a watchdog: if the polling task dies (e.g. proxy / network blip
        causes python-telegram-bot's retry loop to terminate), we re-initialize
        the Application and restart polling. Without this the process stays
        alive but Telegram input goes silent forever.
        """
        self._running = True
        await self._build_and_start_app()
        logger.info("Telegram bot started")

        # Watchdog: every 15s check two independent liveness signals:
        #   1. updater.running flag — catches clean termination of the polling task
        #   2. last successful getUpdates timestamp — catches httpx socket wedge
        #      where the task is "running" but no network IO is actually happening
        # The flag alone is insufficient: we observed a 10+ hour outage where
        # updater.running stayed True while getUpdates went silent.
        backoff = 5.0
        max_backoff = 120.0
        while self._running:
            await asyncio.sleep(15)
            if not self._running:
                break
            try:
                flag_ok = bool(self.app and self.app.updater and self.app.updater.running)
            except Exception:
                flag_ok = False

            stall = None
            if self._get_updates_request is not None:
                stall = time.monotonic() - self._get_updates_request.last_seen_at
            io_ok = stall is None or stall <= GET_UPDATES_STALL_THRESHOLD_SEC

            if flag_ok and io_ok:
                backoff = 5.0  # reset on healthy state
                continue

            reason = (
                f"flag={flag_ok}, last_getUpdates={stall:.1f}s ago"
                if stall is not None
                else f"flag={flag_ok}, no IO timestamp yet"
            )
            logger.warning(
                f"Telegram polling unhealthy ({reason}); attempting restart "
                f"(backoff={backoff}s)"
            )
            try:
                await self._teardown_app(quiet=True)
                await asyncio.sleep(backoff)
                if not self._running:
                    break
                await self._build_and_start_app()
                logger.info("Telegram polling restarted")
                backoff = 5.0
                # Replay anything the agent tried to send while we were down.
                drained = await self._drain_outbox()
                if drained:
                    logger.info(f"Drained {drained} buffered message(s) after restart")
            except Exception as e:
                logger.error(f"Telegram restart failed: {e}")
                backoff = min(backoff * 2, max_backoff)

    async def _build_and_start_app(self) -> None:
        """(Re)create the Application and start polling. Idempotent enough for restart."""
        # Mirror ApplicationBuilder._build_request(get_updates=True) defaults
        # (pool size 1, http/1.1) — only the class is swapped so we get the
        # last_seen_at timestamp without changing PTB's timeout semantics.
        self._get_updates_request = _StallTrackedRequest(
            connection_pool_size=1,
            http_version="1.1",
            proxy=self.proxy or None,
        )
        builder = (
            Application.builder()
            .token(self.bot_token)
            .get_updates_request(self._get_updates_request)
        )
        if self.proxy:
            # Covers the main bot API request pool. getUpdates traffic gets the
            # proxy via the request object above instead, because PTB forbids
            # combining get_updates_proxy() with get_updates_request().
            builder = builder.proxy(self.proxy)
        self.app = builder.build()

        self.app.add_handler(CommandHandler("start", self._start_command))
        self.app.add_handler(CommandHandler("help", self._help_command))
        self.app.add_handler(CommandHandler("status", self._status_command))
        self.app.add_handler(CommandHandler("skills", self._skills_command))
        self.app.add_handler(CommandHandler("compact", self._compact_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        await self.app.initialize()
        await self.app.start()
        from config import get_config

        cfg = get_config()
        default_lang = resolve_ui_language(cfg.ui_language, agent_language=cfg.language)
        await self.app.bot.set_my_commands(bot_commands(default_lang))
        await self.app.bot.set_my_commands(bot_commands("en"), language_code="en")
        await self.app.bot.set_my_commands(bot_commands("zh"), language_code="zh")
        logger.info("Bot commands registered")
        await self.app.updater.start_polling()

    async def _teardown_app(self, quiet: bool = False) -> None:
        """Tear down the current Application (best-effort) so we can rebuild."""
        if not self.app:
            return
        for op_name, op in (
            ("updater.stop", lambda: self.app.updater.stop()),
            ("app.stop", lambda: self.app.stop()),
            ("app.shutdown", lambda: self.app.shutdown()),
        ):
            try:
                await op()
            except Exception as e:
                if not quiet:
                    logger.warning(f"{op_name} during teardown: {e}")
        self.app = None
        # Drop the reference; the rebuild path will install a fresh tracked
        # request. Clearing here avoids the watchdog observing a stale
        # last_seen_at against an already-shutdown HTTPXRequest.
        self._get_updates_request = None

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        await self._teardown_app(quiet=True)
        logger.info("Telegram bot stopped")

    def _is_ready(self) -> bool:
        """True iff the Application + Updater are alive enough to actually send."""
        try:
            return bool(
                self.app
                and self.app.updater
                and self.app.updater.running
            )
        except Exception:
            return False

    def health_status(self) -> str:
        if self._is_ready():
            return "healthy"
        return "starting" if self._running else "stopped"

    async def send(self, text: str, user_id: Optional[str] = None) -> None:
        """Send a message to Telegram. Buffers when polling is being restarted
        so messages produced during the watchdog window aren't lost."""
        target_chat_id = user_id or self.chat_id
        max_length = 4096
        chunks = self._split_message(text, max_length)

        if not self._is_ready():
            # Buffer for later drain. Drop oldest if full so the most recent
            # state of the world wins (better than blocking the agent).
            for chunk in chunks:
                if len(self._outbox) == self._outbox.maxlen:
                    logger.warning("Telegram outbox full; dropping oldest queued message")
                self._outbox.append((chunk, target_chat_id))
            logger.info(
                f"Telegram not ready; buffered {len(chunks)} chunk(s) "
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
        """Send a document to Telegram, buffering during polling restarts."""
        target_chat_id = user_id or self.chat_id
        file_path = Path(path)
        if not file_path.is_file():
            logger.warning(f"Telegram file send skipped; file missing: {file_path}")
            fallback = f"{caption or 'File send failed'}\n\nFile does not exist: `{file_path}`"
            await self.send(fallback, target_chat_id)
            return

        if not self._is_ready():
            if len(self._file_outbox) == self._file_outbox.maxlen:
                logger.warning("Telegram file outbox full; dropping oldest queued file")
            self._file_outbox.append((str(file_path), target_chat_id, caption))
            logger.info(
                f"Telegram not ready; buffered file '{file_path.name}' "
                f"(file_outbox={len(self._file_outbox)})"
            )
            return

        await self._send_document_now(file_path, target_chat_id, caption)

    async def _send_chunks_now(self, chunks, target_chat_id: str) -> None:
        """Send LLM-originated chunks as plain text.

        Telegram command handlers still use Markdown for fixed UI text. Free-form
        LLM output is less predictable, so keep this path plain and lightly strip
        presentation-only Markdown markers before sending.
        """
        for chunk in chunks:
            plain_chunk = _plain_text_for_telegram(chunk)
            try:
                await self.app.bot.send_message(
                    chat_id=target_chat_id,
                    text=plain_chunk,
                )
            except Exception as err:
                logger.error(f"Telegram plain-text send failed: {err}")
                if len(self._outbox) < self._outbox.maxlen:
                    self._outbox.append((chunk, target_chat_id))

    async def _send_document_now(
        self,
        path: Path,
        target_chat_id: str,
        caption: Optional[str] = None,
    ) -> None:
        """Send one document immediately; fall back to a path message on failure."""
        try:
            with path.open("rb") as document:
                await self.app.bot.send_document(
                    chat_id=target_chat_id,
                    document=document,
                    filename=path.name,
                    caption=caption,
                )
        except Exception as err:
            logger.error(f"Telegram document send failed for {path}: {err}")
            fallback = f"{caption or 'File send failed'}\n\nReport file: `{path}`"
            await self.send(fallback, target_chat_id)

    async def _drain_outbox(self) -> int:
        """Flush the outboxes now that the Application is back up. Returns count sent."""
        async with self._drain_lock:
            if (not self._outbox and not self._file_outbox) or not self._is_ready():
                return 0
            sent = 0
            # Snapshot then clear so newly-queued messages during the drain don't
            # interleave (they'll get drained on the next watchdog tick).
            pending = list(self._outbox)
            self._outbox.clear()
            for chunk, chat_id in pending:
                await self._send_chunks_now([chunk], chat_id)
                sent += 1
            pending_files = list(self._file_outbox)
            self._file_outbox.clear()
            for path, chat_id, caption in pending_files:
                await self._send_document_now(Path(path), chat_id, caption)
                sent += 1
            return sent

    def _split_message(self, text: str, max_length: int) -> list:
        """Split long message at paragraph boundaries."""
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

    async def _typing_keepalive(self, chat, stop_event: asyncio.Event):
        """Background task: send typing status every 4 seconds until stop_event is set."""
        try:
            while not stop_event.is_set():
                await chat.send_action("typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing keepalive error (ignored): {e}")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(t("telegram.start", self._ui_lang(update)), parse_mode="Markdown")

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(t("telegram.help", self._ui_lang(update)), parse_mode="Markdown")

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command through the shared command implementation."""
        from channel_commands import CommandScope, build_status_text

        lang = self._ui_lang(update)
        chat_id = str(update.effective_chat.id)
        thread_id = (
            str(update.message.message_thread_id)
            if getattr(update.message, "message_thread_id", None) is not None
            else None
        )
        scope = CommandScope(
            channel="telegram",
            user_id=str(update.effective_user.id),
            conversation_id=chat_id,
            thread_id=thread_id,
        )
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
            logger.error(f"/status failed: {e}")
            await update.message.reply_text(t("command.status.read_failed", lang))
            return
        await update.message.reply_text(status_text, parse_mode="Markdown")

    async def _skills_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /skills command."""
        from skills_loader import get_skills_loader

        lang = self._ui_lang(update)
        loader = get_skills_loader()
        skills = loader.get_all_skills()

        if not skills:
            await update.message.reply_text(t("telegram.skills.empty", lang), parse_mode="Markdown")
            return

        lines = [t("telegram.skills.title", lang, count=len(skills))]
        for name, meta in skills.items():
            triggers = ", ".join(meta.triggers[:3]) if meta.triggers else t("telegram.skills.no_triggers", lang)
            lines.append(f"• `{name}` - {triggers}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _compact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compact command through the shared command implementation."""
        from channel_commands import CommandScope, compact_conversation

        lang = self._ui_lang(update)
        await update.message.reply_text(t("command.compact.running", lang))
        try:
            chat_id = str(update.effective_chat.id)
            thread_id = (
                str(update.message.message_thread_id)
                if getattr(update.message, "message_thread_id", None) is not None
                else None
            )
            scope = CommandScope(
                channel="telegram",
                user_id=str(update.effective_user.id),
                conversation_id=chat_id,
                thread_id=thread_id,
            )
            result = await compact_conversation(scope=scope, lang=lang)
        except Exception as e:
            logger.error(f"/compact failed: {e}")
            await update.message.reply_text(t("command.compact.failed", lang, error=e))
            return
        await update.message.reply_text(result)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user message."""
        user_message = update.message.text
        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)
        thread_id = (
            str(update.message.message_thread_id)
            if getattr(update.message, "message_thread_id", None) is not None
            else None
        )

        logger.info(f"Received message from {user_id}: {user_message}")

        # Start typing keepalive
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            self._typing_keepalive(update.message.chat, stop_typing)
        )

        # Create incoming message
        message = IncomingMessage(
            text=user_message,
            channel="telegram",
            user_id=user_id,
            conversation_id=chat_id,
            reply_to_id=str(update.message.message_id),
            thread_id=thread_id,
            metadata={
                "chat_id": chat_id,
                "sender_id": user_id,
                "username": update.effective_user.username,
                "message_id": update.message.message_id,
                "chat_type": getattr(update.effective_chat, "type", None),
            },
        )

        try:
            # Handle message
            response = await self.message_handler(message)

            # Stop typing
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

            if is_no_action_response(response):
                logger.warning(
                    "Suppressing internal NO_ACTION final response for Telegram user message"
                )
                return

            # Send response
            await self.send(response, chat_id)

        except Exception as e:
            # Stop typing on error
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

            logger.error(f"Error handling message: {e}")
            await update.message.reply_text(
                t("telegram.message.error", self._ui_lang(update))
            )

    def _ui_lang(self, update: Update) -> str:
        from config import get_config

        cfg = get_config()
        channel_language = getattr(getattr(update, "effective_user", None), "language_code", None)
        return resolve_ui_language(
            cfg.ui_language,
            channel_language=channel_language,
            agent_language=cfg.language,
        )
