"""Start-of-turn digest of autonomous project progress.

Closes the user-facing half of the autonomy loop. Phase 2.95 (and the
``manage_projects`` / HTTP advance paths) advance projects in the
background, durably updating the project event log — but nothing told the
*user*. This builds a concise "while you were away" digest from the
project events since the last digest, flagging the items that now need
the user's input, and is surfaced as a header on the next chat turn.

Watermark-gated on the monotonic event id (clock-free), so each batch of
progress is shown exactly once; the first run baselines silently so no
historical backlog is dumped. Pure-ish + fail-safe: a digest failure must
never break a turn.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

# Event types written by core.project_advancer.advance_once.
_ADVANCE_EVENTS = ("autoadvance_step",)
_NEEDS_USER_EVENTS = ("autoadvance_needs_user", "human_gate_triggered")
# Written by ProjectStore._maybe_rollup_project_status when task updates
# settle the whole project's status.
_ROLLUP_EVENTS = ("project_auto_rollup",)
_RELEVANT = frozenset(_ADVANCE_EVENTS + _NEEDS_USER_EVENTS + _ROLLUP_EVENTS)

# DONE/FAILED project lists grow without bound, and only the most recently
# updated few can hold events newer than the watermark — cap the terminal
# scan (``list_projects`` orders by ``updated_at DESC``).
_RECENT_TERMINAL_LIMIT = 10


@dataclass
class DigestResult:
    advanced: int = 0
    projects_touched: int = 0
    needs_user: List[Tuple[str, str]] = field(default_factory=list)  # (project_title, task_desc)
    finished: List[Tuple[str, str]] = field(default_factory=list)  # (project_title, new_status)
    new_event_id: int = 0

    @property
    def has_content(self) -> bool:
        return self.advanced > 0 or bool(self.needs_user) or bool(self.finished)


def summarize_since(store, last_event_id: int, *, per_project_limit: int = 50) -> DigestResult:
    """Scan projects' events newer than ``last_event_id`` and tally
    autonomous progress. The scan covers NEEDS_USER and recent DONE/FAILED
    projects alongside ACTIVE ones: the advancer's tick rolls the project's
    status in the same batch that logs its digest-worthy events, so an
    ACTIVE-only scan hid exactly the batch where the user's input became
    needed. ``new_event_id`` advances past every event scanned (so the next
    call doesn't re-show them), even types we don't surface."""
    res = DigestResult(new_event_id=int(last_event_id))
    candidates: List[dict] = []
    for status, cap in (("ACTIVE", None), ("NEEDS_USER", None),
                        ("DONE", _RECENT_TERMINAL_LIMIT),
                        ("FAILED", _RECENT_TERMINAL_LIMIT)):
        try:
            projs = store.list_projects(status) or []
        except Exception as e:
            logger.debug("digest list_projects(%s) failed: %s", status, e)
            continue
        candidates.extend(projs if cap is None else projs[:cap])
    touched = set()
    for p in candidates:
        pid = p.get("id")
        title = str(p.get("title") or pid or "project")[:40]
        try:
            events = store.list_events(pid, limit=per_project_limit)
        except Exception:
            continue
        for ev in events:
            try:
                eid = int(ev.get("id") or 0)
            except Exception:
                eid = 0
            if eid <= last_event_id:
                continue
            res.new_event_id = max(res.new_event_id, eid)
            etype = ev.get("type")
            if etype not in _RELEVANT:
                continue
            if etype in _ADVANCE_EVENTS:
                touched.add(pid)
                res.advanced += 1
            elif etype in _ROLLUP_EVENTS:
                # Only terminal rollups are digest-worthy; a NEEDS_USER
                # rollup already surfaces via its needs-user event.
                new_status = str(
                    (ev.get("payload") or {}).get("new_status") or ""
                ).upper()
                if new_status in ("DONE", "FAILED"):
                    touched.add(pid)
                    res.finished.append((title, new_status))
            else:  # needs-user
                touched.add(pid)
                payload = ev.get("payload") or {}
                desc = str(payload.get("description") or "")
                if not desc and ev.get("task_id"):
                    try:
                        t = store.get_task(ev["task_id"]) or {}
                        desc = str(t.get("description") or "")
                    except Exception:
                        desc = ""
                res.needs_user.append((title, (desc or "(task)")[:80]))
    res.projects_touched = len(touched)
    return res


def render_digest(res: DigestResult, *, max_needs_user: int = 3) -> str:
    """Render the digest as a short markdown header. Empty when there's
    nothing new worth surfacing."""
    if not res.has_content:
        return ""
    lines = [
        f"**While you were away** — I advanced {res.advanced} task(s) "
        f"on {res.projects_touched} project(s) on my own."
    ]
    if res.finished:
        lines.append(f"{len(res.finished)} project(s) reached a final status:")
        for title, status in res.finished[:max_needs_user]:
            lines.append(f"  - [{title}] → {status}")
    if res.needs_user:
        lines.append(f"{len(res.needs_user)} now need your input:")
        for title, desc in res.needs_user[:max_needs_user]:
            lines.append(f"  - [{title}] {desc}")
        extra = len(res.needs_user) - max_needs_user
        if extra > 0:
            lines.append(f"  - …and {extra} more")
    lines.append("(Ask me, or run `manage_projects action=status`, for details.)")
    return "\n".join(lines)


def load_watermark(path) -> Optional[int]:
    """Return the saved last-seen event id, or ``None`` when the file is
    absent (first run) so the caller can baseline silently."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        return int(json.loads(p.read_text()).get("last_event_id", 0))
    except Exception:
        return None


def save_watermark(path, last_event_id: int) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_event_id": int(last_event_id)}))
        os.replace(tmp, p)
    except Exception as e:
        logger.debug("digest watermark save failed: %s", e)


__all__ = [
    "DigestResult",
    "summarize_since",
    "render_digest",
    "load_watermark",
    "save_watermark",
]
