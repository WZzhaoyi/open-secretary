"""External CLI subagent runner.

The runner treats Codex and Claude Code as local sidecars. It never reads or
copies their auth files; each CLI is responsible for its own login state,
permissions, skills, and rate limits.
"""

import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from config import get_config
from guardrails import BASE_DIR, truncate_output

SubAgentEngine = Literal["codex", "claude"]


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
    """Run supported local sub-agent CLIs in a bounded subprocess."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or BASE_DIR).resolve()

    def is_available(self, engine: str) -> bool:
        if engine not in ("codex", "claude"):
            return False
        return shutil.which(engine) is not None

    def choose_engine(self, requested: Optional[str] = None) -> str:
        """Return a supported engine, preferring explicit request then Claude."""
        if requested:
            engine = requested.lower().strip()
            if engine not in ("codex", "claude"):
                raise ValueError("engine must be 'codex' or 'claude'")
            if not self.is_available(engine):
                raise RuntimeError(f"{engine} CLI is not installed or not on PATH")
            return engine

        preferred = get_config().subagent.default_engine.lower().strip() or "claude"
        order = [preferred, "claude", "codex"]
        for engine in dict.fromkeys(order):
            if self.is_available(engine):
                return engine
        raise RuntimeError("Neither claude nor codex CLI is installed or on PATH")

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
        raise ValueError("engine must be 'codex' or 'claude'")

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
        output_path = self._make_output_path() if engine == "codex" else None
        command = self.build_command(engine, prompt, output_last_message=output_path)

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
