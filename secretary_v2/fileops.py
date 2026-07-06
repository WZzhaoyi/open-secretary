"""Shared text-file editing core for memory_* and file_* tools.

The memory tools and the generic file tools are the same editing model —
content-anchored str_replace with a uniqueness requirement, atomic writes,
and line-numbered snippets — differing only in path guardrails, prompts,
and policies (scaffold/capacity/backup for memory.md). This module holds
the shared mechanics; tool wrappers in runtime.py add the policy layers.

Pure text transforms raise ValueError with a model-facing message; wrappers
decide how to phrase recovery hints (e.g. "Call memory_view" vs "Call
file_read").
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple


def atomic_write_text(path: Path, content: str) -> None:
    """Write a text file via temp file + os.replace so a crash mid-write can
    never leave a torn/half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def str_replace_unique(
    text: str, old_str: str, new_str: str, *, filename: str
) -> Tuple[str, int]:
    """Replace old_str with new_str, requiring exactly one verbatim match.

    Returns (updated_text, changed_line_index). Raises ValueError with a
    model-facing message when old_str matches zero or multiple locations —
    the uniqueness requirement is the protection: the caller must quote the
    current on-disk text before anything is changed.
    """
    count = text.count(old_str)
    if count == 0:
        raise ValueError(f"old_str did not appear verbatim in {filename}.")
    if count > 1:
        matching_lines = [
            str(text[: m.start()].count("\n") + 1)
            for m in re.finditer(re.escape(old_str), text)
        ]
        raise ValueError(
            f"old_str appears {count} times in {filename} "
            f"(lines {', '.join(matching_lines)}). "
            "Add surrounding context so it matches exactly once."
        )

    pos = text.find(old_str)
    changed_line_index = text[:pos].count("\n")
    return text.replace(old_str, new_str), changed_line_index


def numbered_lines(lines: List[str], start_line: int) -> List[str]:
    return [f"{str(start_line + i).rjust(5)}\t{line}" for i, line in enumerate(lines)]


def edit_snippet(updated: str, changed_line_index: int) -> str:
    """Line-numbered context around an edit, echoed back so the model can
    verify the change without a follow-up view/read call."""
    lines = updated.split("\n")
    start = max(0, changed_line_index - 2)
    end = min(len(lines), changed_line_index + 3)
    return "\n".join(numbered_lines(lines[start:end], start + 1))
