"""Channel abstraction for secretary v2."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class IncomingMessage:
    """Incoming message from a channel."""
    text: str
    channel: str  # "telegram" | "cli" | "http" | "scheduled"
    user_id: str
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
        """Send a message to the channel."""
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
