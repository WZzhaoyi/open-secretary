"""Guardrails for secretary v2 - simplified version."""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

BASE_DIR = Path(__file__).parent
PERMISSIONS_FILE = BASE_DIR / "permissions" / "core.yaml"

# Permission policy
DEFAULT_PATH_POLICY = {
    "deny_absolute_paths": True,
    "deny_parent_traversal": True,
    "allowed_read_dirs": [".", "data", "logs", "permissions", "skills", "research", "subagent_runs"],
    "allowed_write_dirs": ["data"],
    "allowed_extensions": [".md", ".yaml", ".yml", ".json", ".txt", ".log", ".csv", ".example"],
    "protected_files": ["config.yaml", "memory.md", "secretary_v2.db", "runtime.py", "main.py", "guardrails.py"],
    "protected_read_files": ["config.yaml", "secretary_v2.db", ".env", ".env.local", ".netrc", ".npmrc", ".pypirc"],
    "deny_dotfiles": True,
    "allowed_dotfiles": [],
    "workspace_exceptions": ["memory.md"],
}

DEFAULT_SHELL_POLICY = {
    "hard_deny_commands": ["mkfs", "dd", "shutdown", "reboot", "halt", "init", "sudo", "su", "doas"],
    "rm_forbidden_args": ["/", "/*", "/.", "/.."],
    "pipe_to_interpreters": ["sh", "bash", "zsh", "fish", "python", "python3", "node", "ruby", "perl"],
    "deny_patterns": [],
}


def _merge_list_policy(defaults: dict, configured: dict) -> dict:
    merged = defaults.copy()
    for key, value in configured.items():
        if isinstance(value, list):
            merged[key] = value
        elif isinstance(value, (bool, str, int, float)) or value is None:
            merged[key] = value
    return merged


def _load_permission_policy() -> dict:
    """Load centralized permission policy, falling back to conservative defaults."""
    try:
        if not PERMISSIONS_FILE.exists():
            return {
                "paths": DEFAULT_PATH_POLICY.copy(),
                "shell": DEFAULT_SHELL_POLICY.copy(),
            }
        raw = yaml.safe_load(PERMISSIONS_FILE.read_text(encoding="utf-8")) or {}
        return {
            "paths": _merge_list_policy(DEFAULT_PATH_POLICY, raw.get("paths") or {}),
            "shell": _merge_list_policy(DEFAULT_SHELL_POLICY, raw.get("shell") or {}),
        }
    except Exception:
        return {
            "paths": DEFAULT_PATH_POLICY.copy(),
            "shell": DEFAULT_SHELL_POLICY.copy(),
        }


PERMISSION_POLICY = _load_permission_policy()
PATH_POLICY = PERMISSION_POLICY["paths"]
SHELL_POLICY = PERMISSION_POLICY["shell"]

# Path security
ALLOWED_READ_DIRS = {str(item) for item in PATH_POLICY.get("allowed_read_dirs", [])}
ALLOWED_WRITE_DIRS = {str(item) for item in PATH_POLICY.get("allowed_write_dirs", [])}
ALLOWED_EXTENSIONS = {str(item) for item in PATH_POLICY.get("allowed_extensions", [])}
PROTECTED_FILES = {str(item) for item in PATH_POLICY.get("protected_files", [])}
PROTECTED_READ_FILES = {str(item) for item in PATH_POLICY.get("protected_read_files", [])}
WORKSPACE_EXCEPTIONS = {str(item) for item in PATH_POLICY.get("workspace_exceptions", [])}
ALLOWED_DOTFILES = {str(item) for item in PATH_POLICY.get("allowed_dotfiles", [])}


@dataclass
class PermissionDecision:
    """Structured permission decision for tool guardrails."""

    allowed: bool
    tool: str
    target: str
    reason: Optional[str] = None
    policy: Optional[str] = None
    allowed_alternative: Optional[str] = None
    message: Optional[str] = None

    def format(self) -> str:
        if self.allowed:
            return "PERMISSION_ALLOWED"
        lines = [
            "PERMISSION_DENIED",
            f"tool: {self.tool}",
            f"reason: {self.reason or 'permission_denied'}",
            f"target: {self.target}",
        ]
        if self.policy:
            lines.append(f"policy: {self.policy}")
        if self.allowed_alternative:
            lines.append(f"allowed_alternative: {self.allowed_alternative}")
        if self.message:
            lines.append(f"message: {self.message}")
        return "\n".join(lines)


def permission_denied(
    tool: str,
    target: str,
    reason: str,
    policy: Optional[str] = None,
    allowed_alternative: Optional[str] = None,
    message: Optional[str] = None,
) -> PermissionDecision:
    return PermissionDecision(
        allowed=False,
        tool=tool,
        target=target,
        reason=reason,
        policy=policy,
        allowed_alternative=allowed_alternative,
        message=message,
    )


def _path_alternative(normalized: str, for_write: bool) -> Optional[str]:
    if _is_memory_path(normalized) and for_write:
        return "memory_update"
    if _is_memory_path(normalized):
        return "memory_read"
    if normalized.startswith(("research/", "subagent_runs/")) and for_write:
        return "start_subagent / subagent artifact manager"
    if normalized == "config.yaml":
        return "ask the user for the specific setting; config.yaml is not readable by agent tools"
    if normalized == "secretary_v2.db":
        return "db_query"
    if for_write:
        return "file_write under data/"
    return None


def _configured_memory_path() -> str:
    try:
        from config import get_config

        cfg = get_config()
        raw = str(getattr(getattr(cfg, "memory", None), "path", "memory.md") or "memory.md")
    except Exception:
        raw = "memory.md"
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(raw)


def _is_memory_path(normalized: str) -> bool:
    return normalized in {"memory.md", _configured_memory_path()}


def check_path_decision(path: str, for_write: bool = False, tool: Optional[str] = None) -> tuple[Optional[str], PermissionDecision]:
    """Check path safety, return (safe_path, structured permission decision)."""
    tool_name = tool or ("file_write" if for_write else "file_read")
    normalized = os.path.normpath(path)

    # Block directory traversal and absolute paths
    if (
        PATH_POLICY.get("deny_parent_traversal", True)
        and normalized.startswith("..")
    ) or (
        PATH_POLICY.get("deny_absolute_paths", True)
        and os.path.isabs(normalized)
    ):
        return None, permission_denied(
            tool_name,
            path,
            "path_escape",
            policy="paths.deny_absolute_paths / paths.deny_parent_traversal",
            message="Path not allowed: no absolute paths or parent directory references",
        )

    # Split directory and file
    parts = normalized.split(os.sep)
    top_dir = parts[0] if len(parts) > 1 else "."
    basename = os.path.basename(normalized)

    if (
        PATH_POLICY.get("deny_dotfiles", True)
        and basename.startswith(".")
        and basename not in ALLOWED_DOTFILES
    ):
        return None, permission_denied(
            tool_name,
            normalized,
            "dotfile_protected",
            policy="paths.deny_dotfiles",
            message="Path not allowed: dotfiles are protected",
        )

    if for_write and (normalized in PROTECTED_FILES or _is_memory_path(normalized)):
        return None, permission_denied(
            tool_name,
            normalized,
            "protected_file",
            policy="paths.protected_files",
            allowed_alternative=_path_alternative(normalized, for_write),
            message=f"{normalized} is a protected file",
        )
    if not for_write and normalized in PROTECTED_READ_FILES:
        return None, permission_denied(
            tool_name,
            normalized,
            "protected_read_file",
            policy="paths.protected_read_files",
            allowed_alternative=_path_alternative(normalized, for_write),
            message=f"{normalized} is a protected read file",
        )

    # Whitelist check
    allowed = ALLOWED_WRITE_DIRS if for_write else ALLOWED_READ_DIRS
    if top_dir not in allowed and normalized not in WORKSPACE_EXCEPTIONS:
        return None, permission_denied(
            tool_name,
            normalized,
            "directory_not_allowed",
            policy="paths.allowed_write_dirs" if for_write else "paths.allowed_read_dirs",
            allowed_alternative=_path_alternative(normalized, for_write),
            message=f"Directory {top_dir} not in whitelist",
        )

    # Extension check
    _, ext = os.path.splitext(normalized)
    if ext and ext not in ALLOWED_EXTENSIONS:
        return None, permission_denied(
            tool_name,
            normalized,
            "extension_not_allowed",
            policy="paths.allowed_extensions",
            message=f"Access to {ext} files not allowed",
        )

    return normalized, PermissionDecision(allowed=True, tool=tool_name, target=normalized)


def check_path(path: str, for_write: bool = False) -> tuple:
    """Check path safety, return (safe_path, error_message)."""
    safe, decision = check_path_decision(path, for_write=for_write)
    return safe, None if decision.allowed else (decision.message or decision.format())

# Shell command safety
SHELL_HARD_DENY_COMMANDS = {
    str(item).lower()
    for item in SHELL_POLICY.get("hard_deny_commands", [])
}
SHELL_RM_FORBIDDEN_ARGS = {
    str(item)
    for item in SHELL_POLICY.get("rm_forbidden_args", [])
}
SHELL_PIPE_TO_INTERPRETER = {
    str(item).lower()
    for item in SHELL_POLICY.get("pipe_to_interpreters", [])
}
SHELL_DENY_PATTERNS = [
    str(item)
    for item in SHELL_POLICY.get("deny_patterns", [])
    if str(item).strip()
]


def check_shell_command_decision(cmd: str, tool: str = "shell") -> PermissionDecision:
    """Check shell command safety, return a structured permission decision."""
    import shlex

    for pattern in SHELL_DENY_PATTERNS:
        if re.search(pattern, cmd, flags=re.IGNORECASE):
            return permission_denied(
                tool,
                cmd,
                "hard_deny_pattern",
                policy="shell.deny_patterns",
                message="命中 shell hard-deny pattern",
            )

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return PermissionDecision(allowed=True, tool=tool, target=cmd)

    if not tokens:
        return PermissionDecision(allowed=True, tool=tool, target=cmd)

    # Check first token
    first = Path(tokens[0]).name.lower().lstrip("-")
    if first in SHELL_HARD_DENY_COMMANDS:
        return permission_denied(
            tool,
            cmd,
            "hard_deny_command",
            policy="shell.hard_deny_commands",
            message=f"禁止执行 {first}",
        )
    if first == "rm":
        for arg in tokens[1:]:
            if arg in SHELL_RM_FORBIDDEN_ARGS:
                return permission_denied(
                    tool,
                    cmd,
                    "rm_forbidden_target",
                    policy="shell.rm_forbidden_args",
                    message="禁止 rm 操作系统根路径",
                )

    # Check pipe targets
    if "|" in cmd:
        try:
            parts = cmd.split("|")
            for part in parts[1:]:
                pipe_tokens = shlex.split(part.strip())
                pipe_target = Path(pipe_tokens[0]).name.lower() if pipe_tokens else ""
                if pipe_target in SHELL_PIPE_TO_INTERPRETER:
                    return permission_denied(
                        tool,
                        cmd,
                        "pipe_to_interpreter",
                        policy="shell.pipe_to_interpreters",
                        message=f"禁止 pipe 到 {pipe_tokens[0]}",
                    )
        except ValueError:
            pass

    return PermissionDecision(allowed=True, tool=tool, target=cmd)


def check_shell_command(cmd: str) -> tuple:
    """Check shell command safety, return (safe, error_message)."""
    decision = check_shell_command_decision(cmd)
    return decision.allowed, None if decision.allowed else (decision.message or decision.format())


# Output truncation
SHELL_MAX_OUTPUT_BYTES = 50 * 1024  # 50KB


def truncate_output(text: str, max_bytes: int = SHELL_MAX_OUTPUT_BYTES) -> str:
    """Truncate output, keeping the tail."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    tail = encoded[-max_bytes:]
    newline = tail.find(b"\n")
    if newline > 0:
        tail = tail[newline + 1:]
    return f"... (truncated, last {max_bytes} bytes) ...\n" + tail.decode("utf-8", errors="replace")
