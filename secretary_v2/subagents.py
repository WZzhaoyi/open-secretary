"""Subagent runner for local CLI sidecars and isolated internal fallback.

The runner treats Codex and Claude Code as local sidecars. It never reads or
copies their auth files; each CLI is responsible for its own login state,
permissions, skills, and rate limits. When those CLIs are unavailable, it can
fall back to a compact internal agent with only configured research skills and
allowlisted bash commands, isolated from the main secretary agent loop, history,
scheduler, and channels.
"""

import asyncio
import fnmatch
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from pydantic_ai import Agent

from config import get_config
from guardrails import BASE_DIR, check_shell_command_decision, truncate_output
from llm_models import build_model
from skills_loader import get_skills_loader

logger = logging.getLogger(__name__)

SubAgentEngine = Literal["codex", "claude", "agent"]
SUPPORTED_ENGINES = ("codex", "claude", "agent")
EXTERNAL_CLI_ENGINES = ("codex", "claude")


def claude_allowed_tools() -> str:
    cfg = get_config().subagent.claude
    tools = list(cfg.allowed_tools)
    tools.extend(_claude_bash_tools(cfg.allowed_bash))
    return ",".join(tool for tool in tools if tool.strip())


def claude_disallowed_tools() -> str:
    cfg = get_config().subagent.claude
    return ",".join(tool for tool in cfg.disallowed_tools if tool.strip())


def _claude_bash_tools(commands: List[str]) -> List[str]:
    tools = []
    for command in commands:
        command = command.strip()
        if not command:
            continue
        tools.append(command if command.startswith("Bash(") else f"Bash({command})")
    return tools


def _build_internal_model():
    """Build an isolated fallback subagent model from the main LLM config."""
    cfg = get_config()
    agent_cfg = cfg.subagent.agent
    model_override = agent_cfg.model.strip() or None
    return build_model(cfg, model_override=model_override)


def _command_allowed_by_patterns(command: str, patterns: List[str]) -> bool:
    command = " ".join((command or "").split())
    return any(fnmatch.fnmatch(command, pattern.strip()) for pattern in patterns if pattern.strip())


def _bash_patterns_from_tools(tools: List[str]) -> List[str]:
    patterns = []
    for tool in tools:
        text = (tool or "").strip()
        if text.startswith("Bash(") and text.endswith(")"):
            pattern = text[len("Bash("):-1].strip()
            if pattern:
                patterns.append(pattern)
    return patterns


def _opencli_search_enabled(tools: List[str]) -> bool:
    return any(pattern.startswith("opencli") for pattern in _bash_patterns_from_tools(tools))


def agent_fallback_missing_binaries() -> List[str]:
    """Binaries named by subagent.agent allowlisted Bash patterns but absent
    from PATH. Non-empty means the fallback agent's shell tools cannot run on
    this host and research degrades to model-only answers."""
    missing = set()
    for pattern in _bash_patterns_from_tools(get_config().subagent.agent.allowed_tools):
        parts = pattern.split()
        first = parts[0] if parts else ""
        # Skip patterns whose command name itself is a wildcard.
        if not first or any(ch in first for ch in "*?["):
            continue
        if shutil.which(first) is None:
            missing.add(first)
    return sorted(missing)


def _internal_agent_system_prompt() -> str:
    cfg = get_config()
    agent_cfg = cfg.subagent.agent
    sections = [agent_cfg.system_prompt.strip()]
    allowed_tools = [tool.strip() for tool in agent_cfg.allowed_tools if tool.strip()]
    disallowed_tools = [tool.strip() for tool in agent_cfg.disallowed_tools if tool.strip()]
    if allowed_tools:
        sections.extend(
            [
                "## Tool Policy",
                "This fallback mirrors the configured Claude `-p` tool allowlist where supported.",
                "Allowed tools:",
                "\n".join(f"- `{tool}`" for tool in allowed_tools),
                "Only `Bash(...)` tools are implemented by the internal fallback.",
                "Commands are still checked by the shared shell hard-deny guardrails before execution.",
            ]
        )
    if disallowed_tools:
        sections.extend(
            [
                "Disallowed tools:",
                "\n".join(f"- `{tool}`" for tool in disallowed_tools),
            ]
        )

    skill_sections = []
    if _opencli_search_enabled(agent_cfg.allowed_tools):
        loader = get_skills_loader()
        max_size = max(1000, int(getattr(cfg.skills, "max_size", 50000) or 50000))
        for name in ("smart-search", "opencli-usage"):
            content = loader.get_skill_content(name)
            if not content:
                continue
            skill_sections.append(f"### {name}\n{content[:max_size]}")
    if skill_sections:
        sections.extend(["## Research Skills", "\n\n".join(skill_sections)])
    return "\n\n".join(section for section in sections if section)


@dataclass
class SubAgentResult:
    engine: str
    prompt: str
    command: List[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary_text(self, max_chars: int = 6000) -> str:
        text = extract_subagent_text(self.engine, self.stdout) or self.stderr.strip()
        return truncate_output(text, max_chars)


def extract_subagent_text(engine: str, stdout: str) -> str:
    """Extract human-readable assistant output from CLI machine formats."""
    if not stdout.strip():
        return ""
    if engine == "claude":
        return _extract_claude_text(stdout)
    if engine == "codex":
        return _extract_codex_text(stdout)
    return stdout.strip()


def _extract_claude_text(stdout: str) -> str:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()
    result = data.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    for key in ("content", "message", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return stdout.strip()


def _extract_codex_text(stdout: str) -> str:
    """Best-effort parser for Codex --json JSONL events.

    The CLI event schema can move over time, so this intentionally looks for
    several common content shapes and uses the last substantial assistant-like
    text rather than depending on a single event type.
    """
    candidates: List[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            candidates.append(line)
            continue
        text = _find_text_in_event(event)
        if text:
            candidates.append(text)
    return candidates[-1].strip() if candidates else stdout.strip()


def _find_text_in_event(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [_find_text_in_event(item) for item in value]
        return "\n".join(piece for piece in pieces if piece).strip()
    if not isinstance(value, dict):
        return ""

    event_type = str(value.get("type") or value.get("event") or "").lower()
    if any(skip in event_type for skip in ("usage", "token", "reasoning_delta")):
        return ""

    for key in ("result", "message", "content", "text", "output", "final_output"):
        if key in value:
            found = _find_text_in_event(value[key])
            if found:
                return found

    role = str(value.get("role") or "").lower()
    if role and role not in ("assistant", "model"):
        return ""

    for key in ("item", "delta", "data", "payload"):
        if key in value:
            found = _find_text_in_event(value[key])
            if found:
                return found
    return ""


class SubAgentRunner:
    """Run supported subagent engines with bounded execution."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or BASE_DIR).resolve()
        self._engine_semaphores = {
            engine: asyncio.Semaphore(self._engine_max_concurrency(engine))
            for engine in EXTERNAL_CLI_ENGINES
        }

    def _engine_max_concurrency(self, engine: str) -> int:
        cfg = get_config().subagent
        if engine == "codex":
            value = cfg.codex.max_concurrency
        elif engine == "claude":
            value = cfg.claude.max_concurrency
        else:
            return 1
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def is_available(self, engine: str) -> bool:
        if engine == "agent":
            return get_config().subagent.agent.enabled
        if engine not in EXTERNAL_CLI_ENGINES:
            return False
        return shutil.which(engine) is not None

    def choose_engine(self, requested: Optional[str] = None) -> str:
        """Return a supported engine, preferring explicit request then Claude."""
        if requested:
            engine = requested.lower().strip()
            if engine not in SUPPORTED_ENGINES:
                raise ValueError("engine must be 'codex', 'claude', or 'agent'")
            if not self.is_available(engine):
                if engine == "agent":
                    raise RuntimeError("internal agent subagent fallback is disabled")
                raise RuntimeError(f"{engine} CLI is not installed or not on PATH")
            if engine == "agent":
                self._warn_agent_fallback_gaps()
            return engine

        cfg = get_config().subagent
        preferred = cfg.default_engine.lower().strip() or "claude"
        fallback = cfg.fallback_engine.lower().strip()
        order = [preferred, "claude", "codex"]
        if fallback:
            order.append(fallback)
        for engine in dict.fromkeys(order):
            if self.is_available(engine):
                if engine == "agent":
                    self._warn_agent_fallback_gaps()
                return engine
        raise RuntimeError(
            "No subagent engine is available: neither claude nor codex CLI is on PATH, "
            "and internal agent fallback is disabled"
        )

    def _warn_agent_fallback_gaps(self) -> None:
        """Surface hosts where the fallback agent is enabled but toothless."""
        missing = agent_fallback_missing_binaries()
        if missing:
            logger.warning(
                "Internal agent fallback selected, but allowlisted command(s) "
                "%s are not on PATH; its shell tools will fail and research "
                "degrades to model-only answers. Install them on this host or "
                "adjust subagent.agent.allowed_tools.",
                ", ".join(missing),
            )

    def build_command(
        self, engine: str, prompt: str, output_last_message: Optional[str] = None
    ) -> List[str]:
        """Build the fixed command line for a subagent sidecar run."""
        if engine == "codex":
            cfg = get_config().subagent.codex
            command = ["codex"]
            for override in cfg.config_overrides:
                if override.strip():
                    command.extend(["-c", override.strip()])
            if cfg.enable_search:
                command.append("--search")
            if cfg.approval_policy.strip():
                command.extend(["--ask-for-approval", cfg.approval_policy.strip()])
            if cfg.model.strip():
                command.extend(["--model", cfg.model.strip()])
            command.extend(["exec", "--json"])
            if cfg.sandbox.strip():
                command.extend(["--sandbox", cfg.sandbox.strip()])
            if output_last_message:
                command.extend(["--output-last-message", output_last_message])
            command.append(prompt)
            return command
        if engine == "claude":
            cfg = get_config().subagent.claude
            command = [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--allowedTools",
                claude_allowed_tools(),
            ]
            disallowed = claude_disallowed_tools()
            if disallowed:
                command.extend(["--disallowedTools", disallowed])
            if cfg.model.strip():
                command.extend(["--model", cfg.model.strip()])
            if cfg.effort.strip():
                command.extend(["--effort", cfg.effort.strip()])
            return command
        if engine == "agent":
            cfg = get_config()
            model = cfg.subagent.agent.model.strip() or cfg.llm.model
            command = ["internal-agent", "--provider", cfg.llm.provider, "--model", model]
            if cfg.llm.effort.strip():
                command.extend(["--effort", cfg.llm.effort.strip()])
            if cfg.subagent.agent.allowed_tools:
                command.extend(["--allowedTools", ",".join(cfg.subagent.agent.allowed_tools)])
            if cfg.subagent.agent.disallowed_tools:
                command.extend(["--disallowedTools", ",".join(cfg.subagent.agent.disallowed_tools)])
            return command
        raise ValueError("engine must be 'codex', 'claude', or 'agent'")

    def resolve_cwd(self, cwd: Optional[str] = None) -> Path:
        if not cwd:
            return self.base_dir
        candidate = (self.base_dir / cwd).resolve()
        if not str(candidate).startswith(str(self.base_dir)):
            raise ValueError("cwd must stay inside the project directory")
        return candidate

    async def run(
        self,
        engine: str,
        prompt: str,
        cwd: Optional[str] = None,
        timeout: int = 1800,
    ) -> SubAgentResult:
        engine = self.choose_engine(engine)
        work_dir = self.resolve_cwd(cwd)
        if engine == "agent":
            return await self._run_internal_agent(prompt, work_dir, timeout)

        output_path = self._make_output_path() if engine == "codex" else None
        command = self.build_command(engine, prompt, output_last_message=output_path)
        semaphore = self._engine_semaphores[engine]
        if semaphore.locked():
            logger.info("Waiting for %s subagent CLI concurrency slot", engine)
        async with semaphore:
            return await self._run_external_cli(
                engine=engine,
                prompt=prompt,
                command=command,
                work_dir=work_dir,
                output_path=output_path,
                timeout=timeout,
            )

    async def _run_external_cli(
        self,
        *,
        engine: str,
        prompt: str,
        command: List[str],
        work_dir: Path,
        output_path: Optional[str],
        timeout: int,
    ) -> SubAgentResult:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate(proc)
            stdout_b, stderr_b = await proc.communicate()
        except asyncio.CancelledError:
            await self._terminate(proc)
            raise

        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        if output_path:
            final_text = self._read_and_unlink(output_path)
            if final_text.strip():
                stdout = final_text

        return SubAgentResult(
            engine=engine,
            prompt=prompt,
            command=command,
            cwd=str(work_dir),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )

    async def _run_internal_agent(
        self,
        prompt: str,
        work_dir: Path,
        timeout: int,
    ) -> SubAgentResult:
        cfg = get_config().subagent.agent
        command = self.build_command("agent", prompt)
        isolated_agent = Agent(
            model=_build_internal_model(),
            system_prompt=_internal_agent_system_prompt(),
        )
        allowed_bash = _bash_patterns_from_tools(cfg.allowed_tools)
        disallowed_bash = _bash_patterns_from_tools(cfg.disallowed_tools)
        shell_timeout = max(1, int(cfg.shell_timeout or 60))

        if allowed_bash:
            @isolated_agent.tool_plain(name="bash")
            async def bash(command: str, timeout: int = shell_timeout) -> str:
                """Run one configured research/search shell command."""
                return await self._run_internal_bash(
                    command=command,
                    work_dir=work_dir,
                    timeout=min(max(1, int(timeout or shell_timeout)), shell_timeout),
                    allowed_patterns=allowed_bash,
                    disallowed_patterns=disallowed_bash,
                )

        try:
            result = await asyncio.wait_for(isolated_agent.run(prompt), timeout)
            stdout = str(result.output)
            return SubAgentResult(
                engine="agent",
                prompt=prompt,
                command=command,
                cwd=str(work_dir),
                exit_code=0,
                stdout=stdout,
                stderr="",
            )
        except asyncio.TimeoutError:
            return SubAgentResult(
                engine="agent",
                prompt=prompt,
                command=command,
                cwd=str(work_dir),
                exit_code=-1,
                stdout="",
                stderr="internal agent subagent timed out",
                timed_out=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return SubAgentResult(
                engine="agent",
                prompt=prompt,
                command=command,
                cwd=str(work_dir),
                exit_code=1,
                stdout="",
                stderr=f"{type(e).__name__}: {e}",
            )

    async def _run_internal_bash(
        self,
        command: str,
        work_dir: Path,
        timeout: int,
        allowed_patterns: List[str],
        disallowed_patterns: Optional[List[str]] = None,
    ) -> str:
        if _command_allowed_by_patterns(command, disallowed_patterns or []):
            return (
                "PERMISSION_DENIED\n"
                "tool: subagent.bash\n"
                "reason: command_disallowed\n"
                f"target: {command}\n"
                "policy: subagent.agent.disallowed_tools"
            )
        if not _command_allowed_by_patterns(command, allowed_patterns):
            return (
                "PERMISSION_DENIED\n"
                "tool: subagent.bash\n"
                "reason: command_not_allowlisted\n"
                f"target: {command}\n"
                "policy: subagent.agent.allowed_tools"
            )

        decision = check_shell_command_decision(command, tool="subagent.bash")
        if not decision.allowed:
            return decision.format()

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout)
            exit_code = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            await self._terminate(proc)
            stdout_b, stderr_b = await proc.communicate()
            exit_code = -1
            return (
                f"Error: command timed out after {timeout}s\n"
                f"exit_code: {exit_code}\n"
                f"stdout:\n{truncate_output((stdout_b or b'').decode('utf-8', errors='replace'))}\n"
                f"stderr:\n{truncate_output((stderr_b or b'').decode('utf-8', errors='replace'))}"
            )

        return (
            f"exit_code: {exit_code}\n"
            f"stdout:\n{truncate_output((stdout_b or b'').decode('utf-8', errors='replace'))}\n"
            f"stderr:\n{truncate_output((stderr_b or b'').decode('utf-8', errors='replace'))}"
        )

    def _make_output_path(self) -> str:
        fd, path = tempfile.mkstemp(prefix="secretary-codex-", suffix=".txt")
        os.close(fd)
        return path

    def _read_and_unlink(self, path: str) -> str:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
        return text

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.2)
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
