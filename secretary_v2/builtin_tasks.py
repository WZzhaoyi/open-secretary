"""Configuration-only built-in scheduled task contracts.

Built-in tasks are deliberately not exposed as agent tools. A task must be
registered in this module and referenced by name from config.yaml before the
scheduler can execute it.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Literal, Mapping, Optional

from memory import Database


BuiltinNotifier = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class BuiltinTaskContext:
    task_id: str
    db: Database
    notify: Optional[BuiltinNotifier] = None


@dataclass(frozen=True)
class BuiltinTaskResult:
    status: Literal["succeeded", "skipped"] = "succeeded"
    details: Dict[str, Any] = field(default_factory=dict)


BuiltinTaskHandler = Callable[[BuiltinTaskContext], Awaitable[BuiltinTaskResult]]


async def _context_maintenance(ctx: BuiltinTaskContext) -> BuiltinTaskResult:
    from maintenance_tasks import context_maintenance

    return await context_maintenance(ctx)


async def _system_health_review(ctx: BuiltinTaskContext) -> BuiltinTaskResult:
    from maintenance_tasks import system_health_review

    return await system_health_review(ctx)


# This static mapping is the only registration path. Imports stay lazy so the
# task implementation can import the contracts above without a module cycle.
BUILTIN_TASKS: Mapping[str, BuiltinTaskHandler] = {
    "context_maintenance": _context_maintenance,
    "system_health_review": _system_health_review,
}
