"""Secretary v2 runtime - Pydantic AI Agent definition + tools."""

import asyncio
import ipaddress
import logging
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider

from compaction import build_summarization_processor, maybe_auto_persist_compact
from config import get_config, SECRETARY_PERSONA, DB_SCHEMA_HINT
from guardrails import (
    BASE_DIR,
    check_path,
    check_path_decision,
    check_shell_command,
    check_shell_command_decision,
    permission_denied,
    truncate_output,
)
from memory import Database

logger = logging.getLogger(__name__)


def _record_agent_event(
    db: Database,
    event_type: str,
    origin: str,
    run_id: Optional[str] = None,
    subject: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort local observability event write.

    Observability must never break the secretary's user-facing behavior.
    """
    try:
        db.create_agent_event(
            event_type=event_type,
            origin=origin,
            run_id=run_id,
            subject=subject,
            payload=payload,
        )
    except Exception as e:
        logger.warning("Failed to record agent event %s: %s", event_type, e)


def _record_permission_denied(deps: "SecretaryDeps", decision) -> str:
    _record_agent_event(
        deps.db,
        "permission_denied",
        origin=deps.origin_channel,
        run_id=deps.run_id or None,
        subject=f"{decision.tool}:{decision.reason}",
        payload={
            "tool": decision.tool,
            "target": decision.target,
            "reason": decision.reason,
            "policy": decision.policy,
            "allowed_alternative": decision.allowed_alternative,
            "message": decision.message,
        },
    )
    return decision.format()


def _local_tz() -> ZoneInfo:
    """Resolve the configured local timezone. Defaults to Asia/Shanghai at config
    load time (see config.py). All LLM-facing time strings flow through this."""
    return ZoneInfo(get_config().timezone)


def _to_local_iso(dt: Optional[datetime], tz: Optional[ZoneInfo] = None) -> str:
    """Render a DB-side datetime (naive UTC) as local-tz ISO with offset suffix.

    DB columns store naive UTC (see memory._utcnow). The LLM sees the result as
    plain text and can't infer the zone, so we always emit an offset-tagged
    string at the injection boundary.
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz or _local_tz()).isoformat()


@dataclass
class SecretaryDeps:
    """Dependencies the agent + tools rely on at run time."""
    db: Database
    origin_channel: str = "cli"
    user_id: str = "default"
    run_id: str = ""
    skill_content: str = ""
    current_time: str = field(
        default_factory=lambda: datetime.now(_local_tz()).isoformat()
    )
    channels: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[Any] = None  # main.Scheduler — typed Any to avoid circular import
    subagent_run_manager: Optional[Any] = None


def build_model(config):
    """Build LLM model based on configuration."""
    provider = config.llm.provider.lower()
    model_name = config.llm.model
    api_key = config.llm.api_key or None
    base_url = config.llm.base_url or None

    if provider == "anthropic":
        if api_key or base_url:
            return AnthropicModel(
                model_name,
                provider=AnthropicProvider(api_key=api_key, base_url=base_url),
            )
        return AnthropicModel(model_name)
    if provider == "openai":
        if api_key or base_url:
            return OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(api_key=api_key, base_url=base_url),
            )
        return OpenAIChatModel(model_name)
    if provider == "gemini":
        if api_key:
            return GeminiModel(model_name, provider=GoogleGLAProvider(api_key=api_key))
        return GeminiModel(model_name)
    raise ValueError(f"Unsupported LLM provider: {provider}")


# Module-level agent. history_processors is wired here so every run benefits
# from compaction. The SummarizationProcessor (from summarization-pydantic-ai)
# transforms the message list in-flight before each run — it summarizes the
# head and keeps a recent tail when the conversation crosses the fraction
# threshold. It does NOT touch the DB; the persisted snapshot is rewritten only
# by the user-triggered /compact (compaction.force_compact).
_config = get_config()
agent = Agent(
    model=build_model(_config),
    deps_type=SecretaryDeps,
    system_prompt=_config.system_prompt or SECRETARY_PERSONA,
    history_processors=[build_summarization_processor()],
)


# Path to the user+agent shared long-term memory scratchpad.
# Free-form markdown that gets injected into every run's system prompt.
MEMORY_FILE = BASE_DIR / "memory.md"
OPEN_EVENTS_CONTEXT_LIMIT = 200


def _language_policy(language: str) -> str:
    normalized = (language or "auto").strip().lower()
    if normalized not in {"zh", "en", "auto"}:
        normalized = "auto"

    if normalized == "en":
        default = "English"
    elif normalized == "auto":
        default = "the user's current language"
    else:
        default = "Chinese"

    return (
        "## Language Policy\n"
        f"- Configured language: `{normalized}`; default user-facing language: {default}.\n"
        "- If the user explicitly asks for a specific language, honor that for the current turn.\n"
        "- Scheduled notifications and proactive messages should use the configured language; "
        "when configured as `auto`, use the language implied by the task prompt and memory.\n"
        "- Do not translate code, file paths, command names, database/table/column names, IDs, "
        "or exact user quotes unless the user asks.\n"
        "- Keep `memory.md` and events concise; preserve user-provided wording when it matters."
    )


def _load_memory_md() -> str:
    """Read memory.md if it exists. Soft-cap to keep system prompt sane."""
    if not MEMORY_FILE.exists():
        return ""
    try:
        text = MEMORY_FILE.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read memory.md: {e}")
        return ""
    # 50KB cap: same as file_read tool, prevents a runaway memory.md from
    # blowing up every prompt.
    return text[:50_000]


@agent.system_prompt
async def dynamic_context(ctx: RunContext[SecretaryDeps]) -> str:
    """Inject schema hint, time, memory.md, attention events, and triggered skills
    on every run so the agent has fresh context without baking them into history.

    Long-term state (preferences, tracked plans/projects) lives in memory.md,
    injected wholesale. Open events are injected as the active attention list.
    Recent events are only a small short-term continuity fragment; precise tasks
    should query the DB.
    """
    deps = ctx.deps
    cfg_root = get_config()
    cfg = cfg_root.history
    tz_name = cfg_root.timezone
    tz = ZoneInfo(tz_name)
    offset = datetime.now(tz).utcoffset() or timedelta(0)
    offset_sec = int(offset.total_seconds())
    sign = "+" if offset_sec >= 0 else "-"
    abs_h, rem = divmod(abs(offset_sec), 3600)
    abs_m = rem // 60
    offset_display = f"{sign}{abs_h:02d}:{abs_m:02d}"
    if abs_m == 0:
        sql_modifier = f"'{sign}{abs_h} hours'"
    else:
        sql_modifier = f"'{sign}{abs_h} hours', '{sign}{abs_m} minutes'"

    # Keep byte-stable prompt material first. DeepSeek's context cache matches
    # exact prefixes, so per-run values like current time and recent events must
    # stay at the tail instead of invalidating the schema/memory/skill prefix.
    stable_parts = [DB_SCHEMA_HINT, _language_policy(cfg_root.language)]
    runtime_parts = []
    open_events_shown = 0
    open_events_total = 0
    recent_events_shown = 0
    recent_events_limit = max(cfg.max_events, 0)

    memory_md = _load_memory_md()
    if memory_md:
        stable_parts.append(f"## 长期记忆 (memory.md)\n{memory_md}")

    try:
        open_events = deps.db.get_events_by_status("open", limit=OPEN_EVENTS_CONTEXT_LIMIT)
        open_events_total = deps.db.count_events_by_status("open")
        open_event_ids = {event.id for event in open_events}
        event_context_parts = []
        if open_events:
            open_events_shown = len(open_events)
            open_text = "\n".join(
                f"- {_to_local_iso(event.created_at, tz)}: "
                f"[id={event.id} type={event.type} status={event.status}] "
                f"{(event.content or '')[:160]}"
                for event in open_events
            )
            truncated_note = (
                "\n注意：open_events 已截断；完整判断必须 db_query 查询 "
                "`SELECT id,type,status,content,created_at FROM events WHERE status='open' ORDER BY created_at DESC`。"
                if open_events_shown < open_events_total
                else ""
            )
            event_context_parts.append(
                "### 待关注事件 open_events\n"
                f"shown {open_events_shown} / total {open_events_total}。"
                "这些是未闭环、需要持有注意力的对象；不要把 recent_events 当作替代。"
                f"{truncated_note}\n"
                f"{open_text}"
            )

        events = deps.db.get_events(limit=cfg.max_events)
        recent_events = [
            event for event in events if event.id not in open_event_ids
        ][:recent_events_limit]
        if recent_events:
            recent_events_shown = len(recent_events)
            events_text = "\n".join(
                f"- {_to_local_iso(event.created_at, tz)}: "
                f"[id={event.id} type={event.type} status={event.status}] "
                f"{(event.content or '')[:100]}"
                for event in recent_events
            )
            event_context_parts.append(
                "### 最近事件片段 recent_events\n"
                f"shown {recent_events_shown} / configured {recent_events_limit}。"
                "这里只是短期连续性片段，已排除上方展示的 open_events；不代表事件全集，"
                "需要完整判断时必须使用 db_query 查询 events 表。\n"
                f"{events_text}"
            )
        if event_context_parts:
            runtime_parts.append("## 事件上下文\n" + "\n\n".join(event_context_parts))
    except Exception as e:
        logger.warning(f"Failed to load recent events: {e}")

    try:
        from skills_loader import get_skills_loader

        loader = get_skills_loader()
        skill_index = loader.get_skill_index()
        if skill_index:
            stable_parts.append(
                "## 可用技能索引\n"
                "下面是可按需加载的项目/用户全局技能索引。索引只说明用途，不等于技能已加载。"
                "当用户请求与某个技能相关，或某个技能明显能提高完成质量时，"
                "先调用 `load_skill(name)` 读取完整技能说明，再继续执行。\n"
                f"{skill_index}"
            )

        auto_skill_parts = []
        max_size = cfg_root.skills.max_size
        for skill_name in loader.get_auto_loaded_skills():
            content = loader.get_skill_content(skill_name)
            if not content:
                continue
            encoded = content.encode("utf-8", errors="replace")
            if len(encoded) > max_size:
                content = encoded[:max_size].decode("utf-8", errors="replace")
                content += f"\n\n... (truncated to {max_size} bytes)"
            auto_skill_parts.append(f"# Skill: {skill_name}\n\n{content}")
        if auto_skill_parts:
            stable_parts.append("## 自动加载技能\n" + "\n\n".join(auto_skill_parts))
    except Exception as e:
        logger.warning(f"Failed to build skill index: {e}")

    if deps.skill_content:
        runtime_parts.append(f"## 已加载技能\n{deps.skill_content}")

    runtime_parts.append(
        "## 当前运行上下文\n"
        f"- now: `{deps.current_time}`\n"
        f"- timezone: `{tz_name}` ({offset_display})\n"
        f"- origin_channel: `{deps.origin_channel}`\n"
        f"- open_events: shown {open_events_shown} / total {open_events_total}\n"
        f"- recent_events: shown {recent_events_shown} / configured {recent_events_limit}\n"
        "- 完整判断必须 db_query 查询数据库；上方事件上下文是提示片段，不是全集。\n\n"
        "### 时间约定\n"
        f"- 显示时间均为 `{tz_name}` ({offset_display})，带 `{offset_display}` 后缀\n"
        "- DB 的 `created_at` / `updated_at` / `last_run` 列存的是 **UTC**（无后缀）；"
        "`db_query` 返回的字符串也是 UTC\n"
        f"- SQL 中要转本地时间用 `datetime(created_at, {sql_modifier})`\n"
        f"- `schedule_task` 的 cron 表达式按 `{tz_name}` 解释（不是 UTC）"
    )

    return "\n\n".join([*stable_parts, *runtime_parts])


# ==================== Skill tools ====================


@agent.tool
async def load_skill(ctx: RunContext[SecretaryDeps], name: str) -> str:
    """Load the full content of a discovered local skill by exact name."""
    try:
        from skills_loader import get_skills_loader

        loader = get_skills_loader()
        skills = loader.get_all_skills()
        if name not in skills:
            available = ", ".join(sorted(skills))
            return f"Error: skill not found: {name}. Available skills: {available}"

        content = loader.get_skill_content(name)
        if not content:
            return f"Error: skill has no readable content: {name}"

        max_size = get_config().skills.max_size
        encoded = content.encode("utf-8", errors="replace")
        if len(encoded) > max_size:
            content = encoded[:max_size].decode("utf-8", errors="replace")
            content += f"\n\n... (truncated to {max_size} bytes)"
        return f"# Skill: {name}\n\n{content}"
    except Exception as e:
        return f"Error loading skill: {e}"


# ==================== Database tools ====================


@agent.tool
async def db_query(ctx: RunContext[SecretaryDeps], sql: str) -> str:
    """Execute a read-only SQL query (SELECT/PRAGMA only)."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith(("SELECT", "PRAGMA", "WITH")):
        return "Error: Only SELECT, PRAGMA, and WITH (CTE) statements are allowed"
    blocked = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH")
    for kw in blocked:
        if kw in sql_upper.split():
            return f"Error: keyword {kw} is not allowed in db_query"
    try:
        results = ctx.deps.db.execute_query(sql)
        if not results:
            return "Query returned no results"
        return str(results)
    except Exception as e:
        return f"Query error: {e}"


@agent.tool
async def db_execute(
    ctx: RunContext[SecretaryDeps], sql: str, params: Optional[List] = None
) -> str:
    """Execute a write SQL statement for ordinary business rows."""
    sql_upper = sql.strip().upper()
    if sql_upper.startswith("SELECT"):
        decision = permission_denied(
            "db_execute",
            "SELECT",
            "wrong_tool",
            policy="db_execute.write_only",
            allowed_alternative="db_query",
            message="Use db_query for SELECT statements",
        )
        return _record_permission_denied(ctx.deps, decision)
    op = sql_upper.split()[0] if sql_upper else ""
    if op in {"DROP", "ATTACH", "DETACH", "ALTER", "CREATE"}:
        decision = permission_denied(
            "db_execute",
            op,
            "blocked_sql_operation",
            policy="db_execute.blocked_operations",
            message=f"{op} operation is blocked",
        )
        return _record_permission_denied(ctx.deps, decision)
    redacted_sql = re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
        "''",
        sql_upper,
    )
    protected_tables = {
        "SCHEDULED_TASKS",
        "MESSAGES",
        "AGENT_EVENTS",
        "SUBAGENT_RUNS",
        "SQLITE_MASTER",
        "SQLITE_SEQUENCE",
    }
    for table in protected_tables:
        if re.search(rf"\b{table}\b", redacted_sql):
            decision = permission_denied(
                "db_execute",
                table.lower(),
                "protected_table",
                policy="db_execute.protected_tables",
                allowed_alternative="dedicated tool or db_query",
                message=f"table {table.lower()} is protected; use the dedicated tool or read-only query",
            )
            return _record_permission_denied(ctx.deps, decision)
    try:
        affected = ctx.deps.db.execute_statement(sql, params or [])
        return f"Statement executed successfully. Rows affected: {affected}"
    except Exception as e:
        return f"Execution error: {e}"


# ==================== File tools ====================


MEMORY_SECTIONS = {
    "User Preferences": "User Preferences",
    "preferences": "User Preferences",
    "user preferences": "User Preferences",
    "profile": "User Preferences",
    "long-term facts": "User Preferences",
    "用户偏好": "User Preferences",
    "偏好": "User Preferences",
    "用户画像": "User Preferences",
    "长期事实": "User Preferences",
    "Collaboration Agreements": "Collaboration Agreements",
    "agreements": "Collaboration Agreements",
    "collaboration agreements": "Collaboration Agreements",
    "协作约定": "Collaboration Agreements",
    "约定": "Collaboration Agreements",
    "Tracked Items": "Tracked Items",
    "tracked items": "Tracked Items",
    "tracking items": "Tracked Items",
    "active topics": "Tracked Items",
    "在追踪的事项": "Tracked Items",
    "追踪事项": "Tracked Items",
    "进行中的主题": "Tracked Items",
}


def _normalize_memory_section(section: str) -> Optional[str]:
    raw = (section or "").strip()
    return MEMORY_SECTIONS.get(raw) or MEMORY_SECTIONS.get(raw.lower())


def _ensure_memory_document(text: str) -> str:
    if text.strip():
        return text
    return (
        "# Long-Term Memory\n\n"
        "> This is a shared note maintained by the user and the secretary agent.\n\n"
        "## User Preferences\n"
        "<!-- Stable preferences, long-term facts, recurring rules, and focus areas. -->\n\n"
        "## Collaboration Agreements\n"
        "<!-- Notification timing, tone, output style, and review rhythm. -->\n\n"
        "## Tracked Items\n"
        "<!-- Plans, projects, positions, opportunities, or topics that need ongoing tracking. -->\n"
    )


def _replace_memory_section(text: str, section: str, body: str) -> str:
    text = _ensure_memory_document(text)
    lines = text.splitlines()
    header = f"## {section}"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break

    if start is None:
        suffix = "\n" if text.endswith("\n") else "\n\n"
        return f"{text}{suffix}{header}\n{body.rstrip()}\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    replacement = [header, *body.rstrip().splitlines()]
    new_lines = [*lines[:start], *replacement, *lines[end:]]
    return "\n".join(new_lines).rstrip() + "\n"


def _append_memory_section(text: str, section: str, content: str) -> str:
    current = _ensure_memory_document(text)
    lines = current.splitlines()
    header = f"## {section}"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break

    entry = content.rstrip()
    if entry and not entry.lstrip().startswith(("-", "*", "1.")):
        entry = f"- {entry}"

    if start is None:
        suffix = "\n" if current.endswith("\n") else "\n\n"
        return f"{current}{suffix}{header}\n{entry}\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    section_lines = lines[start + 1 : end]
    normalized_existing = {line.strip() for line in section_lines if line.strip()}
    if entry.strip() in normalized_existing:
        return current.rstrip() + "\n"

    insert_at = end
    if insert_at > start + 1 and lines[insert_at - 1].strip():
        lines.insert(insert_at, entry)
    else:
        lines.insert(insert_at, entry)
    return "\n".join(lines).rstrip() + "\n"


@agent.tool
async def memory_read(ctx: RunContext[SecretaryDeps]) -> str:
    """Read the complete memory.md file for inspection or complex edits.

    Long-term memory is already injected into every run. Use this when the user
    asks to inspect memory.md, or before complex rewrites that need the latest
    on-disk file. For routine memory updates, prefer memory_update.
    """
    return _load_memory_md() or "memory.md is empty or missing"


@agent.tool
async def memory_update(
    ctx: RunContext[SecretaryDeps],
    section: str,
    content: str,
    mode: str = "append",
) -> str:
    """Update memory.md as the long-term memory store.

    Use this instead of file_write for durable user preferences, collaboration
    agreements, long-term facts, tracking items, trading plans, and topics that
    should affect future reminders or reviews. Supported sections:
    User Preferences, Collaboration Agreements, Tracked Items. Chinese legacy
    section names are also accepted. mode='append' adds one item; mode='replace_section'
    replaces the whole section body.
    """
    canonical = _normalize_memory_section(section)
    if canonical is None:
        return (
            "Error: unknown memory section. Use one of: User Preferences, "
            "Collaboration Agreements, Tracked Items"
        )
    if mode not in ("append", "replace_section"):
        return "Error: mode must be 'append' or 'replace_section'"

    try:
        current = MEMORY_FILE.read_text(encoding="utf-8") if MEMORY_FILE.exists() else ""
        if mode == "replace_section":
            updated = _replace_memory_section(current, canonical, content)
        else:
            updated = _append_memory_section(current, canonical, content)
        MEMORY_FILE.write_text(updated, encoding="utf-8")
        _record_agent_event(
            ctx.deps.db,
            "memory_update",
            origin=ctx.deps.origin_channel,
            run_id=ctx.deps.run_id or None,
            subject=canonical,
            payload={
                "mode": mode,
                "content_chars": len(content or ""),
            },
        )
        return f"memory.md updated: {canonical} ({mode})"
    except Exception as e:
        return f"Error updating memory.md: {e}"


@agent.tool
async def file_read(ctx: RunContext[SecretaryDeps], path: str) -> str:
    """Read a file (whitelist-restricted, max 50KB)."""
    safe, decision = check_path_decision(path, for_write=False, tool="file_read")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)
    file_path = BASE_DIR / safe
    if not file_path.exists():
        return f"Error: File not found: {path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(50_000)
        if len(content) == 50_000:
            content += "\n... [truncated at 50KB]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@agent.tool
async def file_write(
    ctx: RunContext[SecretaryDeps],
    path: str,
    content: str,
    mode: str = "overwrite",
) -> str:
    """Write to a file (whitelist-restricted)."""
    safe, decision = check_path_decision(path, for_write=True, tool="file_write")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)
    file_path = BASE_DIR / safe
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(file_path, write_mode, encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# ==================== HTTP tool ====================


def _is_url_blocked(url: str) -> Optional[str]:
    """Return error message if url targets a private/internal address, else None."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        return f"Invalid URL: {e}"
    host = parsed.hostname
    if not host:
        return "Missing hostname"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"DNS resolution failed: {e}"
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return f"Internal/private address blocked: {ip}"
    return None


@agent.tool
async def http_request(
    ctx: RunContext[SecretaryDeps],
    method: str,
    url: str,
    headers: Optional[Dict] = None,
    body: Optional[str] = None,
) -> str:
    """Send an HTTP request (blocks private IPs via DNS resolution)."""
    import httpx

    block = _is_url_blocked(url)
    if block:
        return f"Error: {block}"

    method_upper = method.upper()
    if method_upper not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"Error: Unsupported method: {method}"

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            resp = await client.request(method_upper, url, headers=headers, content=body)
        return f"Status: {resp.status_code}\nBody: {resp.text[:10_000]}"
    except Exception as e:
        return f"HTTP error: {e}"


# ==================== Search tool ====================


@agent.tool
async def web_search(
    ctx: RunContext[SecretaryDeps], query: str, max_results: int = 5
) -> str:
    """Search the web (Tavily backend; falls back to DDG instant-answer if no key)."""
    cfg = get_config()
    if cfg.search.backend == "tavily" and cfg.search.tavily_api_key:
        return await _search_tavily(query, max_results, cfg.search.tavily_api_key)
    return await _search_duckduckgo(query, max_results)


async def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
    data = resp.json()
    results = [
        f"- [{item.get('title', '')}]({item.get('url', '')}): {item.get('content', '')[:200]}"
        for item in data.get("results", [])
    ]
    answer = data.get("answer", "")
    if answer:
        return f"Answer: {answer}\n\nSources:\n" + "\n".join(results)
    return f"Search results for '{query}':\n" + "\n".join(results) if results else "No results"


async def _search_duckduckgo(query: str, max_results: int) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
        )
    data = resp.json()
    results: List[str] = []
    if data.get("AbstractText"):
        results.append(f"- {data.get('Heading', '')}: {data['AbstractText'][:200]}")
    for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append(f"- {topic['Text'][:200]}")
    if not results:
        return f"No DDG instant answer for '{query}' (configure Tavily API key for real search)"
    return f"Search results for '{query}':\n" + "\n".join(results)


# ==================== Channel tool ====================


@agent.tool
async def send_message(
    ctx: RunContext[SecretaryDeps], text: str, channel: Optional[str] = None
) -> str:
    """Send a message to the user via the origin channel (or an explicit one).

    Keep Telegram/scheduled notification text plain: use short paragraphs and
    simple hyphen bullets, not Markdown headings, bold, tables, or code fences.
    """
    target = channel or ctx.deps.origin_channel
    channel_obj = ctx.deps.channels.get(target)
    if channel_obj is None:
        # For scheduled origin, route to the configured default outgoing channel.
        if target == "scheduled":
            default = get_config().channels.default_outgoing
            channel_obj = ctx.deps.channels.get(default)
            target = default
    if channel_obj is None:
        return f"Error: Channel '{target}' not available"

    # Only forward user_id when it's a real peer address. For scheduled-task
    # runs, ctx.deps.user_id is the literal "scheduler" (a logical identity,
    # not a routable chat_id) — passing it to Telegram caused "Chat not found"
    # errors. In that case, let the channel use its own default recipient.
    forward_user_id = ctx.deps.user_id
    if ctx.deps.origin_channel == "scheduled":
        forward_user_id = None

    try:
        await channel_obj.send(text, forward_user_id)
        _record_agent_event(
            ctx.deps.db,
            "send_message",
            origin=ctx.deps.origin_channel,
            run_id=ctx.deps.run_id or None,
            subject=target,
            payload={
                "target_channel": target,
                "text_chars": len(text or ""),
                "preview": (text or "")[:160],
                "user_id_forwarded": forward_user_id is not None,
            },
        )
        return f"Message sent via {target}"
    except Exception as e:
        return f"Error sending message: {e}"


# ==================== Schedule tool ====================


@agent.tool
async def schedule_task(
    ctx: RunContext[SecretaryDeps],
    action: str,
    task_id: Optional[str] = None,
    cron: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Manage scheduled tasks. Mutations go through the live scheduler so
    new/changed jobs fire immediately without restart."""
    db = ctx.deps.db
    sched = ctx.deps.scheduler

    if action == "list":
        tasks = db.get_scheduled_tasks(enabled_only=False)
        if not tasks:
            return "No scheduled tasks found"
        return "\n".join(
            f"- {t.id}: {t.cron} ({'enabled' if t.enabled else 'disabled'}, "
            f"{'protected' if t.protected else 'normal'})"
            for t in tasks
        )

    if action == "create":
        if not task_id or not cron or not prompt:
            return "Error: task_id, cron, and prompt are required for create"
        try:
            db.create_scheduled_task(task_id, cron, prompt)
            if sched is not None:
                sched.add_job(task_id, cron, prompt)
            return f"Task '{task_id}' created"
        except Exception as e:
            return f"Error creating task: {e}"

    if action == "update":
        if not task_id:
            return "Error: task_id is required for update"
        updates: Dict[str, Any] = {}
        if cron:
            updates["cron"] = cron
        if prompt:
            updates["prompt"] = prompt
        if not updates:
            return "Error: at least one of cron or prompt is required for update"
        try:
            task = db.update_scheduled_task(task_id, **updates)
            if not task:
                return f"Task '{task_id}' not found"
            if sched is not None:
                sched.update_job(task_id, cron=cron, prompt=prompt)
            return f"Task '{task_id}' updated"
        except Exception as e:
            return f"Error updating task: {e}"

    if action == "delete":
        if not task_id:
            return "Error: task_id is required for delete"
        try:
            ok = db.delete_scheduled_task(task_id)
            if not ok:
                return f"Task '{task_id}' not found or is protected"
            if sched is not None:
                sched.remove_job(task_id)
            return f"Task '{task_id}' deleted"
        except Exception as e:
            return f"Error deleting task: {e}"

    return f"Error: unknown action '{action}'. Valid actions: list, create, update, delete"


# ==================== Research tools ====================


@agent.tool
async def start_research(
    ctx: RunContext[SecretaryDeps],
    topic: str,
    engine: Optional[str] = None,
) -> str:
    """Start a non-blocking deep research job using local codex or claude.

    Use this for trading opportunities, industry analysis, or broad questions
    that need multi-step web research outside the main conversation loop.
    `engine` may be "codex" or "claude"; omit it to use the best available CLI.
    Only call this when the user is asking to start a new research job. Never
    call it for status/progress/cancel/list requests about an existing job id.
    """
    manager = ctx.deps.subagent_run_manager
    if manager is None:
        return "Error: subagent run manager is not available"
    try:
        job_id = manager.start(
            input_payload={"topic": topic, "language": get_config().language},
            subject=topic,
            engine=engine,
            origin_channel=ctx.deps.origin_channel,
            user_id=None if ctx.deps.origin_channel == "scheduled" else ctx.deps.user_id,
        )
        chosen = ctx.deps.db.get_subagent_run(job_id)
        engine_text = chosen.engine if chosen else (engine or "auto")
        stages = getattr(getattr(manager, "definition", None), "stages", [])
        stage_text = "/".join(stages) if stages else "multiple stages"
        return (
            f"Started research job `{job_id}` with engine `{engine_text}`.\n"
            f"It will run in the background through {stage_text}; "
            "I will notify you when it finishes."
        )
    except Exception as e:
        return f"Error starting research: {type(e).__name__}: {e}"


@agent.tool
async def get_research_status(ctx: RunContext[SecretaryDeps], job_id: Optional[str] = None) -> str:
    """Get one research job status, or list recent jobs when job_id is omitted.

    Use this for questions like "check research_xxx status" or "list recent research jobs".
    Do not start a new job unless the user explicitly asks for a new research.
    """
    manager = ctx.deps.subagent_run_manager
    if manager is None:
        return "Error: subagent run manager is not available"
    if job_id:
        return manager.status_text(job_id)
    return manager.list_text()


@agent.tool
async def cancel_research(ctx: RunContext[SecretaryDeps], job_id: str) -> str:
    """Cancel a running background research job."""
    manager = ctx.deps.subagent_run_manager
    if manager is None:
        return "Error: subagent run manager is not available"
    return (
        f"Research job `{job_id}` cancellation requested"
        if manager.cancel(job_id)
        else f"Research job `{job_id}` cannot be cancelled or does not exist"
    )


# ==================== Shell tool ====================

# Shell is needed for skills like futuapi/longbridge that drive CLIs and helper
# scripts. Safety: first-token classification + pipe-target check (guardrails),
# subprocess in its own session for clean group kill, hard timeout, output cap.
SHELL_RATE_WINDOW = 60.0
SHELL_RATE_LIMIT = 20
_shell_calls: List[float] = []


def _resolve_shell_cwd(cwd: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if cwd is None or not cwd.strip():
        return str(BASE_DIR), None
    normalized = os.path.normpath(cwd)
    if os.path.isabs(normalized) or normalized.startswith(".."):
        return None, "cwd must stay inside secretary_v2"
    resolved = (BASE_DIR / normalized).resolve()
    try:
        resolved.relative_to(BASE_DIR.resolve())
    except ValueError:
        return None, "cwd must stay inside secretary_v2"
    return str(resolved), None


@agent.tool
async def shell(
    ctx: RunContext[SecretaryDeps],
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> str:
    """Run a shell command. Returns stringified result (exit_code/stdout/stderr or error)."""
    now = time.time()
    _shell_calls[:] = [t for t in _shell_calls if t > now - SHELL_RATE_WINDOW]
    if len(_shell_calls) >= SHELL_RATE_LIMIT:
        decision = permission_denied(
            "shell",
            command,
            "rate_limited",
            policy="shell.rate_limit",
            message="shell rate limit exceeded (20/min)",
        )
        return _record_permission_denied(ctx.deps, decision)
    _shell_calls.append(now)

    decision = check_shell_command_decision(command, tool="shell")
    if not decision.allowed:
        return _record_permission_denied(ctx.deps, decision)

    work_dir, cwd_err = _resolve_shell_cwd(cwd)
    if cwd_err:
        decision = permission_denied(
            "shell",
            cwd or "",
            "cwd_escape",
            policy="shell.cwd_within_base_dir",
            message=cwd_err,
        )
        return _record_permission_denied(ctx.deps, decision)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = proc.communicate()
            return (
                f"Error: command timed out after {timeout}s\n"
                f"exit_code: -1\n"
                f"stdout:\n{truncate_output(stdout or '')}\n"
                f"stderr:\n{truncate_output(stderr or '')}"
            )
        return (
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{truncate_output(stdout or '')}\n"
            f"stderr:\n{truncate_output(stderr or '')}"
        )
    except Exception as e:
        return f"Error: {e}"


# ==================== run_agent ====================


# LLM call retry policy. Re-throws non-transient errors immediately so we don't
# mask bad-request / auth issues behind retries.
_LLM_MAX_ATTEMPTS = 3
_LLM_BACKOFF_BASE_SEC = 2.0

# Most recent run's provider-reported token usage, surfaced by /status. A plain
# module global is safe: runs are serialized through the task_queue, so there's
# no concurrent writer. None until the first run completes.
_last_usage: Optional[Dict[str, Any]] = None


def _usage_details(usage: Any) -> Dict[str, Any]:
    details = getattr(usage, "details", None)
    return details if isinstance(details, dict) else {}


def _build_usage_payload(
    usage: Any,
    *,
    origin_channel: str,
    at: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize Pydantic AI usage plus DeepSeek's OpenAI-compatible extras."""
    details = _usage_details(usage)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cache_read_tokens = getattr(usage, "cache_read_tokens", 0) or 0
    cache_write_tokens = getattr(usage, "cache_write_tokens", 0) or 0
    prompt_cache_hit_tokens = details.get("prompt_cache_hit_tokens")
    prompt_cache_miss_tokens = details.get("prompt_cache_miss_tokens")

    cache_hit_tokens = (
        prompt_cache_hit_tokens
        if isinstance(prompt_cache_hit_tokens, (int, float))
        else cache_read_tokens
    )
    if isinstance(prompt_cache_miss_tokens, (int, float)):
        cache_miss_tokens = prompt_cache_miss_tokens
    elif isinstance(input_tokens, (int, float)):
        cache_miss_tokens = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    else:
        cache_miss_tokens = None

    cache_denom = (
        cache_hit_tokens + cache_miss_tokens
        if isinstance(cache_miss_tokens, (int, float))
        else input_tokens
    )
    cache_hit_ratio = (
        cache_hit_tokens / cache_denom
        if isinstance(cache_denom, (int, float)) and cache_denom > 0
        else None
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": getattr(usage, "requests", None),
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "cache_hit_ratio": cache_hit_ratio,
        "details": details,
        "at": at or datetime.now(_local_tz()).isoformat(timespec="seconds"),
        "origin": origin_channel,
    }


def get_last_usage() -> Optional[Dict[str, Any]]:
    """Return the last completed run's token usage (input/output/total/requests
    + timestamp + origin channel), or None if no run has finished yet.

    `input_tokens` is the real context size the provider billed for the last
    turn — exact, zero extra cost (it rode back on the API response)."""
    return _last_usage


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True if the error is worth retrying — network, timeout, 5xx, rate limit."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                        httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    # pydantic_ai wraps some errors; inspect message as a soft fallback
    msg = str(exc).lower()
    return any(s in msg for s in ("connection error", "timeout", "temporar", "503", "502", "504"))


async def run_agent(
    user_text: str,
    db: Database,
    origin_channel: str = "cli",
    user_id: str = "default",
    skill_content: str = "",
    channels: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    subagent_run_manager: Optional[Any] = None,
) -> str:
    """Run the agent, threading prior conversation in via message_history.

    Persistence flow (this keeps long-running context continuous):
      1. Load recent pydantic-ai messages from DB, bounded by a token budget.
      2. Pass as message_history so the agent sees the running conversation.
      3. After the run, persist result.new_messages() so the next call sees
         the full trajectory (user prompt + tool calls + assistant reply).
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    deps = SecretaryDeps(
        db=db,
        origin_channel=origin_channel,
        user_id=user_id,
        run_id=run_id,
        skill_content=skill_content,
        current_time=datetime.now(_local_tz()).isoformat(),
        channels=channels or {},
        scheduler=scheduler,
        subagent_run_manager=subagent_run_manager,
    )
    _record_agent_event(
        db,
        "run_started",
        origin=origin_channel,
        run_id=run_id,
        subject=user_text[:80],
        payload={
            "user_text_chars": len(user_text or ""),
            "skill_content_loaded": bool(skill_content),
            "user_id": user_id,
        },
    )

    history = db.load_pydantic_messages()
    logger.debug(f"[run_agent] loaded {len(history)} prior messages from DB")

    last_exc: Optional[BaseException] = None
    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        try:
            result = await agent.run(
                user_text,
                deps=deps,
                message_history=history,
            )
            break
        except Exception as e:
            last_exc = e
            if attempt >= _LLM_MAX_ATTEMPTS or not _is_transient_llm_error(e):
                logger.error(
                    f"Agent run failed (attempt {attempt}/{_LLM_MAX_ATTEMPTS}, "
                    f"transient={_is_transient_llm_error(e)}): {e}"
                )
                _record_agent_event(
                    db,
                    "run_failed",
                    origin=origin_channel,
                    run_id=run_id,
                    subject=type(e).__name__,
                    payload={
                        "attempt": attempt,
                        "max_attempts": _LLM_MAX_ATTEMPTS,
                        "transient": _is_transient_llm_error(e),
                        "error": str(e),
                    },
                )
                raise
            backoff = _LLM_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                f"Agent run transient failure (attempt {attempt}/{_LLM_MAX_ATTEMPTS}): "
                f"{type(e).__name__}: {e}; retrying in {backoff}s"
            )
            await asyncio.sleep(backoff)
    else:
        # Unreachable due to break/raise above, but keeps type checker happy.
        raise last_exc  # type: ignore[misc]

    usage_payload: Dict[str, Any] = {}
    try:
        usage = result.usage()
        global _last_usage
        _last_usage = _build_usage_payload(usage, origin_channel=origin_channel)
        usage_payload = dict(_last_usage)
        logger.info(
            "[run_agent] usage input=%s output=%s total=%s requests=%s cache_read=%s cache_write=%s cache_hit=%s cache_miss=%s",
            _last_usage["input_tokens"],
            _last_usage["output_tokens"],
            _last_usage["total_tokens"],
            _last_usage["requests"],
            _last_usage["cache_read_tokens"],
            _last_usage["cache_write_tokens"],
            _last_usage["cache_hit_tokens"],
            _last_usage["cache_miss_tokens"],
        )
    except Exception as e:
        logger.debug(f"[run_agent] usage unavailable: {e}")

    response = str(result.output)
    _record_agent_event(
        db,
        "run_finished",
        origin=origin_channel,
        run_id=run_id,
        subject=response[:80],
        payload={
            "response_chars": len(response),
            "no_action": "NO_ACTION" in response.strip().upper(),
            "usage": usage_payload,
        },
    )

    # Startup self-test: exercise the full agent.run path (including
    # history_processor dispatch — that's the bug class this guards against)
    # but never persist. Reaching this line at all means the pipeline works.
    if origin_channel == "self_test":
        logger.info(f"[run_agent] self-test ok (output preview: {response[:60]!r})")
        return response

    # NO_ACTION: don't persist this turn — used by scheduled tasks that found
    # nothing to report, so they don't pollute history.
    if "NO_ACTION" in response.strip().upper():
        _record_agent_event(
            db,
            "scheduled_no_action" if origin_channel == "scheduled" else "no_action",
            origin=origin_channel,
            run_id=run_id,
            subject=user_text[:80],
            payload={"response_preview": response[:160]},
        )
        # Lift to INFO for scheduled origin so log readers can see the task fired
        # and chose to stay silent (vs. failing). Other origins stay debug.
        if origin_channel == "scheduled":
            logger.info(f"[run_agent] scheduled task → NO_ACTION (silent skip)")
        else:
            logger.debug("[run_agent] NO_ACTION — skipping persistence")
        return response

    try:
        new_msgs = result.new_messages()
        db.save_pydantic_messages(new_msgs)
        logger.debug(f"[run_agent] persisted {len(new_msgs)} new messages")
    except Exception as e:
        logger.error(f"[run_agent] failed to persist messages: {e}")

    try:
        outcome = await maybe_auto_persist_compact(db)
        if outcome and outcome.changed and not outcome.failed:
            target = origin_channel
            channel_obj = (channels or {}).get(target)
            if target == "scheduled" or channel_obj is None:
                target = get_config().channels.default_outgoing
                channel_obj = (channels or {}).get(target)
            if channel_obj is not None:
                await channel_obj.send(
                    "Conversation history was automatically compacted: "
                    f"{outcome.before_messages} -> {outcome.after_messages} messages, "
                    f"{outcome.before_tokens:,} -> {outcome.after_tokens:,} tokens.",
                    None if origin_channel == "scheduled" else user_id,
                )
    except Exception as e:
        logger.error(f"[run_agent] auto persist compaction failed: {e}")

    return response
