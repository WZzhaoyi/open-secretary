"""CLI channel for secretary v2 - REPL and single execution mode."""

import asyncio
import sys
from typing import Optional, Callable, Awaitable

from .base import Channel, IncomingMessage


class CLIChannel(Channel):
    """CLI channel for development and testing."""

    name = "cli"

    def __init__(
        self,
        message_handler: Callable[[IncomingMessage], Awaitable[str]],
        single_message: Optional[str] = None,
    ):
        self.message_handler = message_handler
        self.single_message = single_message
        self._running = False

    async def start(self) -> None:
        """Start the CLI channel."""
        self._running = True

        if self.single_message:
            # Single execution mode
            await self._handle_single_message()
        else:
            # REPL mode
            await self._run_repl()

    async def stop(self) -> None:
        """Stop the CLI channel."""
        self._running = False

    async def send(self, text: str, user_id: Optional[str] = None) -> None:
        """Send a message to stdout."""
        print(f"\nSecretary: {text}\n")

    async def _handle_single_message(self) -> None:
        """Handle a single message and exit."""
        message = IncomingMessage(
            text=self.single_message,
            channel="cli",
            user_id="cli_user",
            conversation_id="cli_user",
        )

        try:
            response = await self.message_handler(message)
            await self.send(response)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    async def _run_repl(self) -> None:
        """Run the REPL loop."""
        print("Secretary v2 CLI - type /quit to exit\n")

        while self._running:
            try:
                # Read input asynchronously
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("> ")
                )

                # Check for quit command
                if user_input.strip().lower() in ("/quit", "/exit", "/q"):
                    print("Goodbye!")
                    break

                # Skip empty input
                if not user_input.strip():
                    continue

                # Create incoming message
                message = IncomingMessage(
                    text=user_input.strip(),
                    channel="cli",
                    user_id="cli_user",
                    conversation_id="cli_user",
                )

                # Handle message
                try:
                    response = await self.message_handler(message)
                    await self.send(response)
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)

            except EOFError:
                # Handle Ctrl+D
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\nGoodbye!")
                break
