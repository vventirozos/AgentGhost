"""Workspace state thread — cross-session persistent state.

Mirrors selfhood.state.SelfStateThread: read-modify-write under a lock,
atomic temp+rename flush, JSON corruption treated as "no prior state".
The bounded slots here are different: a list of tracked files, the
last session timestamp, and the seen-URL dedup set.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from .schema import (
    FileSnapshot,
    TrackedFile,
    WorkspaceState,
    _utcnow_iso,
)

logger = logging.getLogger("GhostWorkspace")

STATE_FILENAME = "state.json"

# Bounds. Tracked files are explicitly user-curated (or auto-promoted)
# so the cap is generous; seen_urls grows unboundedly otherwise — cap
# it FIFO so a long-running agent doesn't accumulate a megabyte of
# stale URLs.
MAX_TRACKED_FILES = 200
MAX_SEEN_URLS = 5000


def _normalise_url(url: str) -> str:
    """Lower-case + strip trailing slash + drop fragment — the same
    sloppy normalisation a human researcher applies when asking
    'did I already read this?'. Not a security boundary, just dedup."""
    if not url:
        return ""
    s = url.strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    if s.endswith("/"):
        s = s[:-1]
    return s.lower()


def _file_snapshot(path: str) -> FileSnapshot:
    """Best-effort stat of ``path``. ``exists=False`` when the file is
    missing — that's load-bearing for "file was deleted since last
    session" detection."""
    try:
        st = os.stat(path)
        return FileSnapshot(
            path=str(path),
            size=int(st.st_size),
            mtime_ns=int(st.st_mtime_ns),
            exists=True,
        )
    except FileNotFoundError:
        return FileSnapshot(path=str(path), exists=False)
    except OSError as e:
        logger.debug("workspace stat failed for %s: %s", path, e)
        return FileSnapshot(path=str(path), exists=False)


def _diff_snapshots(
    prev: Optional[FileSnapshot], curr: FileSnapshot,
) -> Optional[str]:
    """Describe the change between two snapshots. None when nothing
    materially changed; otherwise a short human-readable note."""
    if prev is None:
        return "newly tracked"
    if not prev.exists and curr.exists:
        return "appeared"
    if prev.exists and not curr.exists:
        return "deleted"
    if not curr.exists:
        return None
    if prev.mtime_ns != curr.mtime_ns or prev.size != curr.size:
        delta = curr.size - prev.size
        sign = "+" if delta >= 0 else ""
        return f"modified ({sign}{delta} bytes)"
    return None


class WorkspaceStateThread:
    """Single-file persisted workspace state."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / STATE_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._state: WorkspaceState = self._read_or_empty()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _read_or_empty(self) -> WorkspaceState:
        if not self.path.exists():
            return WorkspaceState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "workspace state read failed (%s); starting empty", e,
            )
            return WorkspaceState()
        try:
            return WorkspaceState.from_dict(data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "workspace state schema mismatch (%s); starting empty", e,
            )
            return WorkspaceState()

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace state flush failed: %s", e)

    # -----------------------------------------------------------------
    # Read API
    # -----------------------------------------------------------------

    @property
    def state(self) -> WorkspaceState:
        return self._state

    def tracked_files(self) -> List[TrackedFile]:
        with self._lock:
            return list(self._state.tracked_files)

    def has_seen_url(self, url: str) -> bool:
        norm = _normalise_url(url)
        if not norm:
            return False
        with self._lock:
            return norm in self._state.seen_urls

    # -----------------------------------------------------------------
    # Write API
    # -----------------------------------------------------------------

    def track_file(
        self, path: str, *, label: str = "",
    ) -> Optional[TrackedFile]:
        path = (path or "").strip()
        if not path:
            return None
        with self._lock:
            for tf in self._state.tracked_files:
                if tf.path == path:
                    if label and not tf.label:
                        tf.label = label
                        self._flush()
                    return tf
            tf = TrackedFile(path=path, label=(label or "").strip())
            self._state.tracked_files.append(tf)
            self._cap(self._state.tracked_files, MAX_TRACKED_FILES)
            self._flush()
            return tf

    def untrack_file(self, path: str) -> bool:
        path = (path or "").strip()
        if not path:
            return False
        with self._lock:
            before = len(self._state.tracked_files)
            self._state.tracked_files = [
                tf for tf in self._state.tracked_files if tf.path != path
            ]
            changed = len(self._state.tracked_files) != before
            if changed:
                self._flush()
            return changed

    def mark_url_seen(self, url: str) -> bool:
        """Record that we've pulled this URL. Returns True if newly
        added, False if already known."""
        norm = _normalise_url(url)
        if not norm:
            return False
        with self._lock:
            if norm in self._state.seen_urls:
                return False
            self._state.seen_urls.append(norm)
            self._cap(self._state.seen_urls, MAX_SEEN_URLS)
            self._flush()
            return True

    def touch_session(self) -> None:
        with self._lock:
            self._state.last_session_at = _utcnow_iso()
            self._flush()

    # -----------------------------------------------------------------
    # File-watching
    # -----------------------------------------------------------------

    def scan_tracked(self) -> List[dict]:
        """Stat every tracked file and return a list of
        ``{path, change, prev, curr}`` dicts describing what changed.
        Side-effect: updates each TrackedFile's ``last_snapshot`` so
        the next scan compares against the freshest state.

        A scan with no changes returns an empty list — that's what the
        caller wants ("nothing to report")."""
        out: List[dict] = []
        with self._lock:
            for tf in self._state.tracked_files:
                curr = _file_snapshot(tf.path)
                change = _diff_snapshots(tf.last_snapshot, curr)
                # Always advance last_seen_at; only emit on actual
                # changes so the wake-up prefix isn't noisy.
                tf.last_seen_at = curr.captured_at
                prev = tf.last_snapshot
                tf.last_snapshot = curr
                if change is not None:
                    out.append({
                        "path": tf.path,
                        "label": tf.label,
                        "change": change,
                        "prev": prev.to_dict() if prev else None,
                        "curr": curr.to_dict(),
                    })
            if out:
                self._flush()
        return out

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _cap(seq, limit: int) -> None:
        overflow = len(seq) - limit
        if overflow > 0:
            del seq[:overflow]

    def format_as_prefix(self, *, max_chars: int = 1200) -> str:
        """Render the current state as a first-person workspace prefix.

        Empty when nothing worth surfacing — a wake-up prefix that says
        only "I'm tracking 0 files" is noise."""
        tracked = self.tracked_files()
        if not (tracked or self._state.last_session_at):
            return ""
        lines: List[str] = []
        if self._state.last_session_at:
            lines.append(
                f"My workspace was last touched on {self._state.last_session_at}.",
            )
        if tracked:
            lines.append(f"I am watching {len(tracked)} file(s):")
            for tf in tracked[-10:]:
                desc = f" ({tf.label})" if tf.label else ""
                lines.append(f"  - {tf.path}{desc}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text
