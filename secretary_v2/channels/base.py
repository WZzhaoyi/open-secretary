"""Channel abstraction for secretary v2."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Any, Optional


def is_no_action_response(text: Optional[str]) -> bool:
    """Return True when the agent returned the internal NO_ACTION marker."""
    if not text:
        return False
    normalized = text.strip().replace("`", "")
    return bool(re.fullmatch(r"(?is)(?:final output:\s*)?NO_ACTION\.?", normalized))


@dataclass
class IncomingMessage:
    """Incoming message from a channel.

    `user_id` is the sender identity. `conversation_id` is the routable chat /
    group / DM target where replies should go. In simple CLI and private-chat
    cases these are often the same value; in group chats they are deliberately
    different.
    """
    text: str
    channel: str  # "telegram" | "feishu" | "cli" | "http" | "scheduled"
    user_id: str
    conversation_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Channel(ABC):
    """Base channel interface."""

    name: str

    @abstractmethod
    async def start(self) -> None:
        """Start the channel."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel."""
        pass

    @abstractmethod
    async def send(self, text: str, user_id: Optional[str] = None) -> None:
        """Send a message to a routable target.

        The parameter remains named `user_id` for backward compatibility, but
        channel implementations should interpret it as a target conversation id
        when their platform distinguishes sender identity from chat identity.
        """
        pass

    async def send_file(
        self,
        path: str | Path,
        caption: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Send a file, or fall back to a text path when unsupported."""
        path_text = str(path)
        text = f"{caption}\n\nFile: `{path_text}`" if caption else f"File: `{path_text}`"
        await self.send(text, user_id=user_id)
