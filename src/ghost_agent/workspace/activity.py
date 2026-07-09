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
import math
import re
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .schema import WorkspaceEvent

logger = logging.getLogger("GhostWorkspace")

ACTIVITY_FILENAME = "activity.jsonl"

# Compaction bounds. The log is append-only and read in full by
# recent()/count()/kinds() on the per-turn prompt-assembly path, so an
# uncapped file makes every turn slower forever. When the file exceeds
# _COMPACT_MAX_BYTES on append, it is rewritten in place keeping the
# newest _COMPACT_KEEP_LINES events.
_COMPACT_MAX_BYTES = 2 * 1024 * 1024
_COMPACT_KEEP_LINES = 2000

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    """Lowercased alnum tokens longer than 2 chars. Splitting on
    non-alnum means paths and URLs decompose into their components, so
    a query like "train.py" matches the event that touched it."""
    return {t for t in _TOKEN_RE.findall(str(text or "").lower()) if len(t) > 2}


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
                self._maybe_compact_locked()
            return self.path
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace activity append failed: %s", e)
            return None

    def _maybe_compact_locked(self) -> None:
        """Rewrite the log keeping only the newest events once it grows
        past the byte cap. Caller must hold ``self._lock``. Best-effort —
        a failed compaction must never fail the append that triggered it."""
        try:
            if self.path.stat().st_size <= _COMPACT_MAX_BYTES:
                return
            with self.path.open("r", encoding="utf-8") as f:
                tail = deque(f, maxlen=_COMPACT_KEEP_LINES)
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.writelines(tail)
            tmp.replace(self.path)
            logger.info(
                "workspace activity log compacted to newest %d events",
                len(tail),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace activity compaction failed: %s", e)

    def iter_events(self) -> Iterator[WorkspaceEvent]:
        """Yield all events, oldest first. Robust to mid-file
        corruption (skips malformed lines).

        The raw lines are read under the lock; parsing and yielding
        happen OUTSIDE it. Holding a non-reentrant lock across ``yield``
        meant an abandoned generator kept the lock until GC, and a
        consumer that appended mid-iteration deadlocked."""
        try:
            with self._lock:
                if not self.path.exists():
                    return
                with self.path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
        except OSError as e:
            logger.warning("workspace activity read failed: %s", e)
            return
        for line in lines:
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

    def recent(
        self, limit: int = 10, *, kind: Optional[str] = None,
    ) -> List[WorkspaceEvent]:
        """Tail of the activity log, optionally filtered by ``kind``."""
        if limit <= 0:
            return []
        # deque(maxlen=) keeps memory at O(limit) instead of
        # materialising every event in the file just to slice the tail.
        items: deque = deque(maxlen=limit)
        for ev in self.iter_events():
            if kind and ev.kind != kind:
                continue
            items.append(ev)
        return list(items)

    def search(self, query: str, *, limit: int = 10) -> List[WorkspaceEvent]:
        """Relevance-ranked keyword search over the whole log — same
        zero-dependency IDF-weighted token overlap as selfhood's
        ``search_my_past``, so a rare, distinctive query term (a
        filename, a project word) dominates a common one. Best match
        first, newest first on ties; no matched token → empty list.

        The log is bounded by compaction (≤ ~2000 events), so a full
        scan per query is cheap and needs no persistent index."""
        if limit <= 0:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        events: List[WorkspaceEvent] = []
        haystacks: List[set] = []
        doc_freq: dict = {}
        for ev in self.iter_events():
            hay = " ".join((
                ev.summary or "",
                ev.kind or "",
                ev.project_id or "",
                " ".join(str(v) for v in (ev.payload or {}).values()),
            ))
            toks = _tokenize(hay)
            events.append(ev)
            haystacks.append(toks)
            for t in toks:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        if not events:
            return []
        n = len(events)
        idf = {
            t: math.log((n + 1) / (doc_freq.get(t, 0) + 1)) + 1.0
            for t in q_tokens
        }
        scored: List[tuple] = []
        for ev, toks in zip(events, haystacks):
            score = sum(idf[t] for t in q_tokens if t in toks)
            if score > 0:
                scored.append((score, ev.timestamp, ev))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [ev for _, _, ev in scored[:limit]]

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
