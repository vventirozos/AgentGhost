"""Reactions to detected workspace file changes (feature 2A).

The wake-up scan already records ``file_changed`` events for every tracked
file that moved, but nothing consumed them. This module is the first
consumer: when a tracked Python file changes, it cheaply checks whether the
file still PARSES, so the agent is told on its next wake that a file it's
working on is currently broken — instead of discovering it mid-task and
burning a turn.

Pure + synchronous (it runs on the prompt-assembly path) and never raises —
a check that itself failed must not block the wake-up prefix. Syntax/parse
errors are the reliably-detectable class without executing the file; true
runtime ImportErrors need execution and are out of scope here (that is what
the optional nearest-test re-run would cover).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("GhostWorkspace")

# Don't try to parse enormous files on the hot path; a generated data blob
# that happens to end in .py isn't worth the latency.
_PARSE_MAX_BYTES = 400_000


def check_changed_python_files(file_changes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return ``[{"path", "error"}]`` for changed ``.py`` files that no
    longer parse. Deleted files and non-Python files are skipped; so is any
    file that can't be read or is too large.
    """
    warnings: List[Dict[str, str]] = []
    seen: set = set()
    for ch in file_changes or []:
        path = str(ch.get("path", ""))
        change = str(ch.get("change", ""))
        if not path.endswith(".py") or "deleted" in change:
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            p = Path(path)
            if not p.is_file():
                continue
            if p.stat().st_size > _PARSE_MAX_BYTES:
                continue
            # utf-8-sig strips a BOM (ast.parse rejects U+FEFF even though
            # the real import machinery strips it — a valid file must not
            # be flagged broken). errors="replace" (not "ignore") keeps
            # invalid bytes visible as U+FFFD so a file the interpreter
            # would REFUSE doesn't silently pass the check; "ignore" also
            # depended on the process locale via read_text's default
            # encoding, mangling non-ASCII source under LC_ALL=C.
            src = p.read_bytes().decode("utf-8-sig", errors="replace")
        except Exception:
            continue
        try:
            ast.parse(src, filename=path)
        except SyntaxError as e:
            warnings.append({
                "path": path,
                "error": f"{type(e).__name__}: {e.msg} (line {e.lineno})",
            })
        except Exception:
            # A non-syntax failure (e.g. null bytes) — note it generically
            # rather than silently dropping a file the agent just edited.
            warnings.append({"path": path, "error": "could not parse file"})
    return warnings
