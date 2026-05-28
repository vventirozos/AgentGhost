"""Workspace activity log — heterogeneous JSONL of WorkspaceEvent.

Same shape and discipline as ``selfhood.autobiographical``: append-only,
sink failures swallowed, lazy iteration so the hot path stays cheap.

Each event is a ``WorkspaceEvent`` with a ``kind`` (file_changed |
task_outcome | research | command | note) and a free-form ``payload``.
One file rather than five so the consumer can iterate the world in
chronological order — useful for the narrative consolidation.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .schema import WorkspaceEvent

logger = logging.getLogger("GhostWorkspace")

ACTIVITY_FILENAME = "activity.jsonl"


class WorkspaceActivity:
    """Append-only workspace event log."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / ACTIVITY_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def append(self, event: WorkspaceEvent) -> Optional[Path]:
        """Write one event. Returns the path on success, None on failure
        or when disabled. Never raises — activity capture is secondary."""
        if not self.enabled:
            return None
        if not event.kind:
            return None
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(event.to_jsonl())
                    f.write("\n")
                    f.flush()
            return self.path
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace activity append failed: %s", e)
            return None

    def iter_events(self) -> Iterator[WorkspaceEvent]:
        """Yield all events, oldest first. Robust to mid-file
        corruption (skips malformed lines)."""
        if not self.path.exists():
            return
        try:
            with self._lock:
                with self.path.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            d = json.loads(s)
                        except json.JSONDecodeError:
                            continue
                        try:
                            yield WorkspaceEvent.from_dict(d)
                        except Exception:  # noqa: BLE001
                            continue
        except OSError as e:
            logger.warning("workspace activity read failed: %s", e)

    def recent(
        self, limit: int = 10, *, kind: Optional[str] = None,
    ) -> List[WorkspaceEvent]:
        """Tail of the activity log, optionally filtered by ``kind``."""
        if limit <= 0:
            return []
        items: List[WorkspaceEvent] = []
        for ev in self.iter_events():
            if kind and ev.kind != kind:
                continue
            items.append(ev)
        return items[-limit:]

    def count(self, *, kind: Optional[str] = None) -> int:
        n = 0
        for ev in self.iter_events():
            if kind and ev.kind != kind:
                continue
            n += 1
        return n

    def kinds(self) -> dict:
        """Return a {kind: count} dict — used by the stats / summary
        renderers to give a sense of "what's in the log"."""
        out: dict = {}
        for ev in self.iter_events():
            out[ev.kind] = out.get(ev.kind, 0) + 1
        return out
