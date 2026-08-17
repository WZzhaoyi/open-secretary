"""Logging helpers."""

import logging
import re


TELEGRAM_BOT_TOKEN_RE = re.compile(r"\bbot(\d+):([A-Za-z0-9_-]+)")

# Matches httpx INFO records for a healthy Telegram long-poll round-trip, e.g.
#   HTTP Request: POST https://api.telegram.org/bot8780891083:***B3ego/getUpdates "HTTP/1.1 200 OK"
# These land once every ~10s and bury everything else in the log once polling
# is established. Non-2xx getUpdates responses and every other Telegram endpoint
# (getMe / setMyCommands / sendMessage / ...) stay visible so connection setup,
# outbound replies, and failures remain observable.
_TELEGRAM_GETUPDATES_OK_RE = re.compile(
    r'^HTTP Request: POST https://api\.telegram\.org/[^"\s]*?/getUpdates "HTTP/1\.1 200 OK"$'
)


def redact_secrets(text: str) -> str:
    """Redact secrets in already-rendered log messages."""
    if not text:
        return text

    def _telegram_repl(match: re.Match) -> str:
        bot_id = match.group(1)
        secret = match.group(2)
        tail = secret[-5:] if len(secret) > 5 else secret
        return f"bot{bot_id}:***{tail}"

    return TELEGRAM_BOT_TOKEN_RE.sub(_telegram_repl, text)


class SecretRedactionFilter(logging.Filter):
    """Redact known secrets from log records before handlers emit them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(record.getMessage())
        record.args = ()
        return True


def install_secret_redaction_filter() -> None:
    """Install the redaction filter on root handlers and the root logger."""
    root = logging.getLogger()
    if not any(isinstance(f, SecretRedactionFilter) for f in root.filters):
        root.addFilter(SecretRedactionFilter())
    for handler in root.handlers:
        if not any(isinstance(f, SecretRedactionFilter) for f in handler.filters):
            handler.addFilter(SecretRedactionFilter())


class TelegramPollingNoiseFilter(logging.Filter):
    """Drop routine Telegram `getUpdates "200 OK"` INFO records from httpx.

    Long-polling fires one such record every ~10s and otherwise crowds the log.
    Only the happy-path polling line is suppressed; abnormal (non-2xx) polls
    and all other endpoints keep their httpx INFO logging so connection
    setup, sends, and error responses stay visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO or record.name != "httpx":
            return True
        return not _TELEGRAM_GETUPDATES_OK_RE.match(record.getMessage())


def install_telegram_polling_noise_filter() -> None:
    """Install the polling-noise filter on root handlers."""
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, TelegramPollingNoiseFilter) for f in handler.filters):
            handler.addFilter(TelegramPollingNoiseFilter())
