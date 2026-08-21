"""Secretary v2 scheduler.

Persistence model: the secretary's `scheduled_tasks` SQL table is the source of
truth for our domain (handler, prompt/task, protected, enabled, execution
status). On startup we sync
ALL DB tasks into APScheduler's in-memory jobstore — not just the ones in
config.yaml — so tasks created at runtime via the schedule_task tool survive
restarts. APScheduler's own jobstore stays in-memory; durability comes from us.
"""

import asyncio
import logging
import uuid
from typing import Awaitable, Callable, Mapping, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from cron_utils import build_cron_trigger
from builtin_tasks import (
    BUILTIN_TASKS,
    BuiltinTaskContext,
    BuiltinTaskHandler,
    BuiltinTaskResult,
)

from channels.base import IncomingMessage
from config import get_config
from memory import Database, _utcnow


logger = logging.getLogger(__name__)


class Scheduler:
    """APScheduler-backed scheduler that takes its task set from the SQLite db."""

    def __init__(
        self,
        db: Database,
        task_handler: Callable[[IncomingMessage], Awaitable[Optional[str]]],
        builtin_tasks: Optional[Mapping[str, BuiltinTaskHandler]] = None,
        builtin_notifier: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.db = db
        self.task_handler = task_handler
        self._config = get_config()
        self._scheduler = AsyncIOScheduler(timezone=self._config.timezone)
        self._builtin_tasks = dict(
            BUILTIN_TASKS if builtin_tasks is None else builtin_tasks
        )
        self._builtin_notifier = builtin_notifier
        self._execution_lock = asyncio.Lock()
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
            if (
                task_cfg.handler == "builtin"
                and task_cfg.task not in self._builtin_tasks
            ):
                raise ValueError(f"unknown builtin task in config: {task_cfg.task!r}")
            if task_id in existing:
                # Refresh in-place so yaml edits take effect on next start.
                self.db.update_scheduled_task(
                    task_id,
                    cron=task_cfg.cron,
                    prompt=task_cfg.prompt,
                    handler=task_cfg.handler,
                    builtin_task=task_cfg.task or None,
                    enabled=1 if task_cfg.enabled else 0,
                    protected=1,
                )
            else:
                self.db.create_scheduled_task(
                    task_id,
                    task_cfg.cron,
                    task_cfg.prompt,
                    protected=True,
                    handler=task_cfg.handler,
                    builtin_task=task_cfg.task or None,
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
                self._add_job_internal(
                    task.id,
                    task.cron,
                    task.prompt,
                    handler=task.handler,
                    builtin_task=task.builtin_task,
                )
            except Exception as e:
                logger.error(f"Failed to load task {task.id}: {e}")

    # ---- public API used by the schedule_task tool ----

    def add_job(self, task_id: str, cron: str, prompt: str) -> None:
        """Add the apscheduler job. Caller is responsible for the DB row."""
        self._add_job_internal(task_id, cron, prompt, handler="agent")
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
        self._add_job_internal(
            task.id,
            task.cron,
            task.prompt,
            handler=task.handler,
            builtin_task=task.builtin_task,
        )
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

    def _add_job_internal(
        self,
        task_id: str,
        cron: str,
        prompt: str,
        *,
        handler: str,
        builtin_task: Optional[str] = None,
    ) -> None:
        if handler not in {"agent", "builtin"}:
            raise ValueError(f"unsupported schedule handler: {handler!r}")
        if handler == "builtin":
            if not builtin_task:
                raise ValueError(f"builtin schedule {task_id!r} has no task name")
            if builtin_task not in self._builtin_tasks:
                raise ValueError(f"unknown builtin task: {builtin_task!r}")
        trigger = build_cron_trigger(cron, self._config.timezone)
        self._scheduler.add_job(
            self._execute_task,
            trigger=trigger,
            id=task_id,
            args=[task_id, prompt, handler, builtin_task],
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    async def _execute_task(
        self,
        task_id: str,
        prompt: str,
        handler: str = "agent",
        builtin_task: Optional[str] = None,
    ):
        async with self._execution_lock:
            return await self._execute_task_serial(
                task_id,
                prompt,
                handler=handler,
                builtin_task=builtin_task,
            )

    async def _execute_task_serial(
        self,
        task_id: str,
        prompt: str,
        *,
        handler: str,
        builtin_task: Optional[str],
    ):
        logger.info("Triggering scheduled task: %s (handler=%s)", task_id, handler)
        attempt_id = f"scheduled_{uuid.uuid4().hex[:12]}"
        attempted_at = _utcnow()
        self.db.update_scheduled_task(
            task_id,
            last_attempt=attempted_at,
            last_error=None,
        )
        self.db.create_agent_event(
            "scheduled_task_started",
            origin="scheduled",
            run_id=attempt_id,
            subject=task_id,
            payload={"task_id": task_id, "handler": handler, "task": builtin_task},
        )
        message = IncomingMessage(
            text=prompt,
            channel="scheduled",
            user_id="scheduler",
            conversation_id=None,
            metadata={
                "task_id": task_id,
                "outgoing": self._config.channels.default_outgoing,
            },
        )
        try:
            result_details = {}
            if handler == "builtin":
                builtin_handler = self._builtin_tasks.get(builtin_task or "")
                if builtin_handler is None:
                    raise RuntimeError(f"unknown builtin task: {builtin_task!r}")
                result = await builtin_handler(
                    BuiltinTaskContext(
                        task_id=task_id,
                        db=self.db,
                        notify=self._builtin_notifier,
                    )
                )
                if not isinstance(result, BuiltinTaskResult):
                    raise RuntimeError(
                        f"builtin task {builtin_task!r} returned an invalid result"
                    )
                outcome = result.status
                result_details = result.details
                response = result
            elif handler == "agent":
                response = await self.task_handler(message)
                if response is None:
                    raise RuntimeError("scheduled agent task returned no result")
                outcome = "succeeded"
            else:
                raise RuntimeError(f"unsupported schedule handler: {handler!r}")

            succeeded_at = _utcnow()
            self.db.update_scheduled_task(
                task_id,
                last_run=succeeded_at,
                last_success=succeeded_at,
                last_error=None,
            )
            self.db.create_agent_event(
                "scheduled_task_succeeded",
                origin="scheduled",
                run_id=attempt_id,
                subject=task_id,
                payload={
                    "task_id": task_id,
                    "handler": handler,
                    "task": builtin_task,
                    "outcome": outcome,
                    "details": result_details,
                },
            )
            logger.info("Task %s done (outcome=%s)", task_id, outcome)
            return response
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.db.update_scheduled_task(task_id, last_error=error)
            self.db.create_agent_event(
                "scheduled_task_failed",
                origin="scheduled",
                run_id=attempt_id,
                subject=task_id,
                payload={
                    "task_id": task_id,
                    "handler": handler,
                    "task": builtin_task,
                    "error": error,
                },
            )
            logger.exception("Task %s failed", task_id)
            return None
