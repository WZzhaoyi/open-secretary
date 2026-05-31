"""Secretary v2 main entry point."""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any

from config import get_config
from memory import Database
from runtime import run_agent
from channels.base import IncomingMessage
from channels.cli_channel import CLIChannel
from channels.telegram_channel import TelegramChannel
from channels.http_channel import HTTPChannel
from logging_utils import install_secret_redaction_filter
from skills_loader import get_skills_loader
from scheduler import Scheduler
from subagent_runs import SubAgentRunManager, parse_subagent_shortcut

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
install_secret_redaction_filter()
# Quiet down APScheduler: python-telegram-bot's JobQueue owns its own
# APScheduler instance that restarts on every Telegram channel restart,
# spamming "Scheduler started/Added job" at INFO. Our own scheduler logs
# under the "scheduler" logger, which stays at INFO.
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RESEARCH_ID_RE = re.compile(r"\bresearch_[0-9a-fA-F]{6,32}\b")
RESEARCH_LIST_WORDS = ("最近研究任务", "研究任务列表", "列出研究任务", "list research")


class SecretaryApp:
    """Secretary v2 application."""

    def __init__(self, channel_type: str = "cli", single_message: str = None):
        self.config = get_config()
        # In single-channel dev modes, force the scheduler/agent to deliver
        # to the active channel so scheduled-task output is visible during
        # development (otherwise it would be lost trying to reach an
        # unconfigured Telegram).
        if channel_type in ("cli", "http"):
            self.config.channels.default_outgoing = channel_type

        self.db = Database()
        self.skills_loader = get_skills_loader()
        self.channel_type = channel_type
        self.single_message = single_message

        # Channels
        self.channels: Dict[str, Any] = {}

        # Initialize scheduler (created BEFORE channels so it can be threaded into deps)
        self.scheduler = Scheduler(
            db=self.db,
            task_handler=self._handle_scheduled_message,
        )
        self.subagent_run_manager = SubAgentRunManager(
            db=self.db,
            notifier=self._send_research_notification,
            agent_name="deep_research",
        )

        self._init_channels()

    def _init_channels(self):
        """Initialize channels based on configuration."""
        # Always create CLI channel
        self.channels["cli"] = CLIChannel(
            message_handler=self._handle_user_message,
            single_message=self.single_message,
        )

        # Predict which channels will actually run so the Telegram /status
        # command can surface the truth ("telegram + http") rather than a
        # hardcoded label.
        if self.channel_type == "cli":
            active = ["cli"]
        elif self.channel_type == "http":
            active = ["http"] if self.config.channels.http.enabled else []
        else:
            active = []
            if self.config.channels.telegram.bot_token:
                active.append("telegram")
            if self.config.channels.http.enabled:
                active.append("http")

        # Create Telegram channel if configured
        if self.config.channels.telegram.bot_token:
            self.channels["telegram"] = TelegramChannel(
                bot_token=self.config.channels.telegram.bot_token,
                chat_id=self.config.channels.telegram.chat_id,
                message_handler=self._handle_user_message,
                peer_channel_names=active,
            )

        # Create HTTP channel if configured
        if self.config.channels.http.enabled:
            self.channels["http"] = HTTPChannel(
                token=self.config.channels.http.token,
                message_handler=self._handle_user_message,
                response_channel=self.channels.get("telegram"),
            )

    def _get_active_channel(self):
        """Get the active channel based on channel_type."""
        if self.channel_type in self.channels:
            return self.channels[self.channel_type]
        return self.channels.get("cli")

    async def _handle_user_message(self, message: IncomingMessage) -> str:
        """Handle incoming user message."""
        try:
            research_shortcut = parse_subagent_shortcut(
                message.text,
                id_pattern=RESEARCH_ID_RE,
                list_words=RESEARCH_LIST_WORDS,
            )
            if research_shortcut:
                action, job_id = research_shortcut
                if action == "status" and job_id:
                    return self.subagent_run_manager.status_text(job_id)
                if action == "cancel" and job_id:
                    ok = self.subagent_run_manager.cancel(job_id)
                    return (
                        f"Research job `{job_id}` cancellation requested"
                        if ok
                        else f"Research job `{job_id}` cannot be cancelled or does not exist"
                    )
                if action == "list":
                    return self.subagent_run_manager.list_text()

            triggered_skills = self.skills_loader.get_triggered_skills(
                message.text,
                include_auto=False,
            )
            skill_content = ""
            if triggered_skills:
                skill_contents = []
                for skill_name in triggered_skills:
                    content = self.skills_loader.get_skill_content(skill_name)
                    if content:
                        skill_contents.append(content)
                skill_content = "\n\n".join(skill_contents)

            response = await run_agent(
                user_text=message.text,
                db=self.db,
                origin_channel=message.channel,
                user_id=message.user_id,
                skill_content=skill_content,
                channels=self.channels,
                scheduler=self.scheduler,
                subagent_run_manager=self.subagent_run_manager,
            )
            return response

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return f"Sorry, an error occurred while processing the message: {e}"

    async def _handle_scheduled_message(self, message: IncomingMessage) -> str:
        """Handle scheduled task message.

        Delivery contract for scheduled-origin runs: the agent MUST use the
        send_message tool to push anything to the user. We deliberately do
        NOT auto-forward the agent's final output as a fallback — that
        fallback caused two failure modes in practice:
          1. The model would emit a polite "no pending reminders" report instead of
             NO_ACTION, and the fallback would dutifully forward the noise.
          2. After calling send_message itself, the model would add a
             trailing "sent" final output, and the fallback would
             double-send it.
        With no fallback, silence is the default; explicit tool call is the
        only delivery path. final output exists only for NO_ACTION detection
        (which gates message persistence in runtime.run_agent).

        Failure path is different from success: if the run itself raises,
        we push a one-line alert to default_outgoing so the user finds out
        in minutes instead of hours (this was the 2026-05-12 incident, where
        every scheduled task silently exception'd inside run_agent for ~6
        hours before being noticed).
        """
        try:
            return await run_agent(
                user_text=message.text,
                db=self.db,
                origin_channel="scheduled",
                user_id="scheduler",
                channels=self.channels,
                scheduler=self.scheduler,
                subagent_run_manager=self.subagent_run_manager,
            )
        except Exception as e:
            task_id = message.metadata.get("task_id", "?") if message.metadata else "?"
            logger.error(f"Error handling scheduled message ({task_id}): {e}")
            await self._send_failure_alert(task_id, e)
            return None

    async def _send_failure_alert(self, task_id: str, exc: BaseException) -> None:
        """Best-effort one-line alert when a scheduled run blows up.

        Bypasses the agent entirely — calls the channel.send primitive
        directly so it survives even when the agent loop is the thing that's
        broken. Any failure here is just logged; we don't want the alert
        path to mask the original error.
        """
        target_name = self.config.channels.default_outgoing
        channel_obj = self.channels.get(target_name)
        if channel_obj is None:
            logger.error(
                f"Cannot send failure alert for {task_id}: "
                f"default_outgoing channel '{target_name}' not in {list(self.channels)}"
            )
            return
        try:
            await channel_obj.send(
                f"⚠️ Scheduled task `{task_id}` failed\n"
                f"{type(exc).__name__}: {exc}\n"
                "See logs at /private/tmp/secretary_v2.log",
                user_id=None,
            )
        except Exception as e2:
            logger.error(f"Failed to deliver failure alert for {task_id}: {e2}")

    async def _send_research_notification(
        self,
        origin_channel: str,
        text: str,
        user_id: str = None,
        artifact_path: str = None,
    ) -> None:
        """Deliver research completion/failure notifications outside agent.run."""
        target_name = origin_channel
        if target_name in ("scheduled", "self_test") or target_name not in self.channels:
            target_name = self.config.channels.default_outgoing
        channel_obj = self.channels.get(target_name)
        if channel_obj is None:
            logger.error(
                f"Cannot send research notification: channel '{target_name}' unavailable"
            )
            return
        try:
            await channel_obj.send(text, user_id)
            if artifact_path:
                await channel_obj.send_file(
                    artifact_path,
                    caption="Full research report",
                    user_id=user_id,
                )
        except Exception as e:
            logger.error(f"Failed to deliver research notification: {e}")

    def _channels_to_start(self):
        """Decide which channels to actually start based on --channel.

        Policy:
        - --channel cli   : only CLI (interactive REPL, dev/single-message mode)
        - --channel http  : only HTTP webhook
        - --channel telegram (or anything else): Telegram polling + HTTP webhook
                            concurrently (CLI is excluded because its REPL
                            blocks stdin)
        """
        if self.channel_type == "cli":
            return [self.channels["cli"]]
        if self.channel_type == "http":
            return [self.channels["http"]] if "http" in self.channels else []
        # production / telegram mode: every non-cli channel that's configured
        return [c for name, c in self.channels.items() if name != "cli"]

    async def _startup_self_test(self) -> None:
        """One synthetic agent.run() to confirm the dispatch path is intact.

        Catches integration regressions before users feel them. The bug class
        this guards against: a code change passes type-checking and import
        but breaks pydantic-ai's runtime introspection (e.g. an annotation
        forward-ref that can't be resolved, a tool registration that fails
        silently). Without this hook, the next user message or scheduled
        task is the first signal — that's how we lost 2026-05-12 morning.

        We pass empty channels + None scheduler so any tool the agent might
        attempt fails harmlessly: we're testing the run pipeline itself, not
        side-effecting tools. Persistence is skipped via origin_channel
        ``"self_test"`` in run_agent.

        On failure: log CRITICAL and re-raise. Caller (run) lets it
        propagate, the finally block stops scheduler, asyncio.run prints
        traceback, process exits non-zero. Loud beats silent here.
        """
        logger.info("Startup self-test: running synthetic agent.run()")
        try:
            await asyncio.wait_for(
                run_agent(
                    user_text="Diagnostic ping. Reply only with OK and do not call any tools.",
                    db=self.db,
                    origin_channel="self_test",
                    user_id="self_test",
                    channels={},
                    scheduler=None,
                    subagent_run_manager=None,
                ),
                timeout=60.0,
            )
            logger.info("Startup self-test: PASSED")
        except Exception as e:
            logger.critical(
                f"Startup self-test FAILED — refusing to start channels. "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def run(self):
        """Run the application: start scheduler + all selected channels concurrently."""
        try:
            await self.scheduler.start()
            await self._startup_self_test()
            resumed_research = self.subagent_run_manager.resume_incomplete()
            if resumed_research:
                logger.info(
                    "Resumed incomplete research jobs: %s",
                    ", ".join(resumed_research),
                )

            channels = self._channels_to_start()
            if not channels:
                logger.error(
                    f"No channels to start for --channel={self.channel_type!r}; "
                    f"configured channels: {list(self.channels.keys())}"
                )
                return

            logger.info(
                f"Starting {len(channels)} channel(s): {[c.name for c in channels]}"
            )

            # Run every channel as its own task so a crash in one doesn't tear
            # down the others. We wait until any task finishes (which usually
            # only happens on stop/error).
            tasks = {
                asyncio.create_task(c.start(), name=f"channel:{c.name}"): c
                for c in channels
            }
            done, pending = await asyncio.wait(
                tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                exc = t.exception()
                if exc:
                    logger.error(f"Channel task {t.get_name()} crashed: {exc}")
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.subagent_run_manager.stop()
            await self.scheduler.stop()
            for channel in self.channels.values():
                try:
                    await channel.stop()
                except Exception as e:
                    logger.error(f"Error stopping channel {channel.name}: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Secretary v2")
    parser.add_argument(
        "--channel",
        choices=["cli", "telegram", "http"],
        default="cli",
        help=(
            "Run mode (default: cli for safety). "
            "cli = interactive REPL only (dev). "
            "telegram = Telegram polling + HTTP webhook concurrently (production). "
            "http = HTTP webhook only."
        ),
    )
    parser.add_argument(
        "--send",
        type=str,
        default=None,
        help="Single message to send and exit (only meaningful with --channel cli)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create and run application
    app = SecretaryApp(
        channel_type=args.channel,
        single_message=args.send,
    )

    asyncio.run(app.run())


if __name__ == "__main__":
    main()
