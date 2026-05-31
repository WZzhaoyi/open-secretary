"""Logging helpers."""

import logging
import re


TELEGRAM_BOT_TOKEN_RE = re.compile(r"\bbot(\d+):([A-Za-z0-9_-]+)")


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
