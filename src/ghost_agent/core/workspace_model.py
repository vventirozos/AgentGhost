# src/ghost_agent/core/workspace_model.py
"""Workspace State Model.

Maintains a structured representation of the sandbox state so the agent
can reason about what has changed, what's missing, and what it already
knows — without re-reading files every turn.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("GhostAgent")


@dataclass
class FileState:
    """Tracked state of a single file in the workspace."""
    path: str
    size: int = 0
    content_hash: str = ""
    file_type: str = ""          # csv, py, json, etc.
    last_modified: float = 0.0
    last_read_turn: int = -1     # Which turn the agent last read this file
    summary: str = ""            # What the agent knows about this file
    columns: List[str] = field(default_factory=list)  # For tabular data
    row_count: int = -1          # For tabular data (-1 = unknown)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size": self.size,
            "file_type": self.file_type,
            "last_modified": self.last_modified,
            "last_read_turn": self.last_read_turn,
            "summary": self.summary,
            "columns": self.columns,
            "row_count": self.row_count,
        }


@dataclass
class WorkspaceDiff:
    """Differences between two workspace states."""
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"Added: {', '.join(self.added[:5])}")
        if self.removed:
            parts.append(f"Removed: {', '.join(self.removed[:5])}")
        if self.modified:
            parts.append(f"Modified: {', '.join(self.modified[:5])}")
        return "; ".join(parts) if parts else "No changes"

    def to_dict(self) -> dict:
        return {"added": self.added, "removed": self.removed, "modified": self.modified}


class WorkspaceModel:
    """Structured model of the agent's sandbox workspace."""

    def __init__(self, sandbox_dir: Optional[Path] = None):
        self.sandbox_dir = sandbox_dir
        self.files: Dict[str, FileState] = {}
        self.current_turn: int = 0
        self._last_scan_time: float = 0.0

    def scan(self, sandbox_dir: Optional[Path] = None) -> WorkspaceDiff:
        """Scan the workspace and return a diff against the previous state.

        This is cheap (just stat calls, no content reads) and should be
        called at the start of each turn.
        """
        scan_dir = sandbox_dir or self.sandbox_dir
        if not scan_dir or not Path(scan_dir).exists():
            return WorkspaceDiff()

        old_paths = set(self.files.keys())
        new_paths: Set[str] = set()
        diff = WorkspaceDiff()

        try:
            for root, dirs, filenames in os.walk(scan_dir):
                # Skip hidden directories and common noise
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in (
                    '__pycache__', 'node_modules', '.git',
                )]
                for fname in filenames:
                    if fname.startswith('.'):
                        continue
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, scan_dir)
                    new_paths.add(rel_path)

                    try:
                        stat = os.stat(full_path)
                    except OSError:
                        continue

                    if rel_path not in self.files:
                        # New file
                        diff.added.append(rel_path)
                        self.files[rel_path] = FileState(
                            path=rel_path,
                            size=stat.st_size,
                            file_type=Path(fname).suffix.lstrip('.'),
                            last_modified=stat.st_mtime,
                        )
                    else:
                        # Check if modified
                        existing = self.files[rel_path]
                        if stat.st_mtime > existing.last_modified or stat.st_size != existing.size:
                            diff.modified.append(rel_path)
                            existing.size = stat.st_size
                            existing.last_modified = stat.st_mtime
                            # Invalidate summary on modification
                            existing.summary = ""
                            existing.columns = []
                            existing.row_count = -1
        except OSError as exc:
            logger.warning("WorkspaceModel scan failed: %s", exc)

        # Detect removed files
        removed = old_paths - new_paths
        for path in removed:
            diff.removed.append(path)
            del self.files[path]

        self._last_scan_time = time.time()
        return diff

    def record_read(self, path: str, summary: str = "",
                    columns: Optional[List[str]] = None,
                    row_count: int = -1):
        """Record that the agent read a file and what it learned."""
        if path in self.files:
            fs = self.files[path]
            fs.last_read_turn = self.current_turn
            if summary:
                fs.summary = summary
            if columns is not None:
                fs.columns = columns
            if row_count >= 0:
                fs.row_count = row_count

    def record_write(self, path: str, summary: str = ""):
        """Record that the agent wrote/modified a file."""
        if path not in self.files:
            self.files[path] = FileState(
                path=path,
                file_type=Path(path).suffix.lstrip('.'),
                last_modified=time.time(),
            )
        fs = self.files[path]
        fs.last_modified = time.time()
        fs.last_read_turn = self.current_turn
        if summary:
            fs.summary = summary

    def what_do_i_know(self, path: str) -> str:
        """Return a summary of what the agent already knows about a file."""
        if path not in self.files:
            return ""
        fs = self.files[path]
        parts = []
        if fs.summary:
            parts.append(f"Summary: {fs.summary}")
        if fs.columns:
            parts.append(f"Columns: {', '.join(fs.columns)}")
        if fs.row_count >= 0:
            parts.append(f"Rows: {fs.row_count}")
        if fs.last_read_turn >= 0:
            turns_ago = self.current_turn - fs.last_read_turn
            parts.append(f"Last read: {turns_ago} turn(s) ago")
        return " | ".join(parts) if parts else ""

    def what_am_i_missing(self, goal: str) -> List[str]:
        """Identify files that might be relevant but haven't been read.

        Uses simple keyword matching between the goal and filenames.
        """
        goal_lower = goal.lower()
        keywords = {w for w in goal_lower.split() if len(w) > 3}
        unread = []
        for path, fs in self.files.items():
            if fs.last_read_turn < 0:  # Never read
                path_lower = path.lower()
                if any(kw in path_lower for kw in keywords):
                    unread.append(path)
        return unread

    def get_recently_modified(self, turns_back: int = 3) -> List[FileState]:
        """Return files modified in the last N turns."""
        min_turn = self.current_turn - turns_back
        return [
            fs for fs in self.files.values()
            if fs.last_read_turn >= min_turn or (
                fs.last_modified > self._last_scan_time - 300  # 5 min
            )
        ]

    def advance_turn(self):
        """Advance the turn counter. Call at the start of each agent turn."""
        self.current_turn += 1

    def get_context_for_prompt(self, max_files: int = 20) -> str:
        """Format workspace state for injection into LLM prompt."""
        if not self.files:
            return ""
        lines = ["### WORKSPACE STATE:"]
        sorted_files = sorted(
            self.files.values(),
            key=lambda f: f.last_modified,
            reverse=True,
        )[:max_files]
        for fs in sorted_files:
            info = f"  {fs.path} ({fs.file_type}, {fs.size}B)"
            if fs.summary:
                info += f" — {fs.summary[:80]}"
            elif fs.last_read_turn < 0:
                info += " [unread]"
            lines.append(info)
        if len(self.files) > max_files:
            lines.append(f"  [... +{len(self.files) - max_files} more files]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "current_turn": self.current_turn,
            "file_count": len(self.files),
            "files": {p: f.to_dict() for p, f in self.files.items()},
        }
