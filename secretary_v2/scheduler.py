"""Secretary v2 scheduler.

Persistence model: the secretary's `scheduled_tasks` SQL table is the source of
truth for our domain (prompt, protected, enabled, last_run). On startup we sync
ALL DB tasks into APScheduler's in-memory jobstore — not just the ones in
config.yaml — so tasks created at runtime via the schedule_task tool survive
restarts. APScheduler's own jobstore stays in-memory; durability comes from us.
"""

import logging
import re
from typing import Awaitable, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from channels.base import IncomingMessage
from config import get_config
from memory import Database, _utcnow

# APScheduler uses 0=Mon..6=Sun for day_of_week (see apscheduler.triggers.cron.
# expressions.WEEKDAYS), while standard Unix cron uses 0=Sun..6=Sat (7 also Sun).
# Without translation, `1-5` silently means Tue-Sat instead of the intended
# Mon-Fri — that exact bug skipped 2026-05-11 (Mon) morning_trend_scan and made
# it fire on Saturdays instead. We translate digits in the day_of_week field
# only (other fields use numbers normally) and only when they're standalone
# weekday tokens (not step values like `*/2`).
_UNIX_DOW_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _translate_unix_dow(field: str) -> str:
    """Map digits 0-7 in a cron day_of_week field to APScheduler symbolic names.

    The negative lookbehind `(?<!/)` skips digits that follow `/` (step values),
    so `*/2` and `1-5/2` are preserved correctly.
    """
    return re.sub(
        r"(?<!/)\b[0-7]\b",
        lambda m: _UNIX_DOW_NAMES[int(m.group(0))],
        field,
    )

logger = logging.getLogger(__name__)


class Scheduler:
    """APScheduler-backed scheduler that takes its task set from the SQLite db."""

    def __init__(
        self,
        db: Database,
        task_handler: Callable[[IncomingMessage], Awaitable[Optional[str]]],
    ):
        self.db = db
        self.task_handler = task_handler
        self._config = get_config()
        self._scheduler = AsyncIOScheduler(timezone=self._config.timezone)
        self._running = False

    # ---- lifecycle ----

    async def start(self):
        await self._sync_config_to_db()
        await self._sync_db_to_scheduler()
        self._scheduler.start()
        self._running = True
        logger.info(
            f"Scheduler started: {len(self._scheduler.get_jobs())} job(s) loaded "
            f"(timezone={self._config.timezone})"
        )

    async def stop(self):
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("Scheduler stopped")

    # ---- sync helpers ----

    async def _sync_config_to_db(self):
        """Make the DB reflect config.yaml for all yaml-declared tasks.

        The `protected` field is the provenance flag that drives this sync:
          - protected=1  → task came from config.yaml (its prompt/cron/enabled
                           are owned by yaml; the schedule_task tool MUST NOT
                           mutate or delete it at runtime).
          - protected=0  → task was created at runtime by the LLM via the
                           schedule_task tool (yaml knows nothing about it).

        Without this distinction we couldn't tell "yaml deleted this task" from
        "LLM created an unrelated task" on restart, so the cleanup pass at the
        end keys off `protected` to avoid wiping user/agent-created jobs.
        """
        existing = {t.id: t for t in self.db.get_scheduled_tasks(enabled_only=False)}
        for task_id, task_cfg in self._config.schedules.items():
            if task_id in existing:
                # Refresh in-place so yaml edits take effect on next start.
                self.db.update_scheduled_task(
                    task_id,
                    cron=task_cfg.cron,
                    prompt=task_cfg.prompt,
                    enabled=1 if task_cfg.enabled else 0,
                    protected=1,
                )
            else:
                self.db.create_scheduled_task(
                    task_id, task_cfg.cron, task_cfg.prompt, protected=True
                )
                logger.info(f"Imported config-declared task: {task_id}")

        # Sweep orphans: rows that *were* yaml-declared (protected=1) but the
        # current yaml no longer mentions them. Without this, deleting a task
        # from yaml would leave the row + cron firing forever. We deliberately
        # leave protected=0 rows alone — those are the LLM's runtime tasks.
        yaml_ids = set(self._config.schedules.keys())
        for task in existing.values():
            if task.protected and task.id not in yaml_ids:
                # force=True is required: delete_scheduled_task otherwise
                # refuses protected rows (the LLM tool's safety net), but
                # here we *are* the authoritative source telling it to go.
                self.db.delete_scheduled_task(task.id, force=True)
                logger.info(f"Removed orphaned config-declared task: {task.id}")

    async def _sync_db_to_scheduler(self):
        """Load every enabled DB task into APScheduler — including runtime-created ones.

        This is the fix for the v1 persistence bug where runtime-created tasks
        only lived in config.yaml and were lost on restart.
        """
        tasks = self.db.get_scheduled_tasks(enabled_only=True)
        for task in tasks:
            try:
                self._add_job_internal(task.id, task.cron, task.prompt)
            except Exception as e:
                logger.error(f"Failed to load task {task.id}: {e}")

    # ---- public API used by the schedule_task tool ----

    def add_job(self, task_id: str, cron: str, prompt: str) -> None:
        """Add the apscheduler job. Caller is responsible for the DB row."""
        self._add_job_internal(task_id, cron, prompt)
        logger.info(f"Scheduled job added: {task_id} ({cron})")

    def update_job(
        self,
        task_id: str,
        cron: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Replace the job in the scheduler with current DB state."""
        tasks = self.db.get_scheduled_tasks(enabled_only=False)
        task = next((t for t in tasks if t.id == task_id), None)
        if task is None:
            logger.warning(f"update_job: task {task_id} not in DB")
            return
        if not task.enabled:
            self.remove_job(task_id)
            return
        self._add_job_internal(task.id, task.cron, task.prompt)
        logger.info(f"Scheduled job updated: {task_id}")

    def remove_job(self, task_id: str) -> None:
        try:
            self._scheduler.remove_job(task_id)
            logger.info(f"Scheduled job removed: {task_id}")
        except Exception as e:
            logger.warning(f"remove_job({task_id}): {e}")

    def get_jobs(self):
        return self._scheduler.get_jobs()

    # ---- internals ----

    def _add_job_internal(self, task_id: str, cron: str, prompt: str) -> None:
        parts = cron.split()
        if len(parts) != 5:
            raise ValueError(f"cron must have 5 fields, got {len(parts)}: {cron!r}")
        dow = _translate_unix_dow(parts[4])
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=dow,
            timezone=self._config.timezone,
        )
        self._scheduler.add_job(
            self._execute_task,
            trigger=trigger,
            id=task_id,
            args=[task_id, prompt],
            replace_existing=True,
        )

    async def _execute_task(self, task_id: str, prompt: str):
        logger.info(f"Triggering scheduled task: {task_id}")
        message = IncomingMessage(
            text=prompt,
            channel="scheduled",
            user_id="scheduler",
            metadata={
                "task_id": task_id,
                "outgoing": self._config.channels.default_outgoing,
            },
        )
        try:
            response = await self.task_handler(message)
            self.db.update_scheduled_task(task_id, last_run=_utcnow())
            logger.info(f"Task {task_id} done")
            return response
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return None
