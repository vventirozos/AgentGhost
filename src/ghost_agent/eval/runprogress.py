"""Progress that is READ, never derived — the contract for long runs.

WHY THIS EXISTS (2026-08-09). A 90-minute bench was launched whose progress
could not be read: its counter went to a block-buffered stdout that would not
flush until exit. Four separate progress figures were then reported to the
operator, and every one was wrong, because each was DERIVED rather than read:

  * "280 trials"      — denominator computed from a buggy pool replica; the
                        bench's own banner said 58 cases, i.e. 464 trials;
  * "99% complete"    — cheap_calls / 2, assuming exactly two judge calls per
                        trial, which is false when a call retries;
  * "64 distinct cases" — a regex whose 70-char window overran short user
                        requests into fault-mutated evidence;
  * several ETAs      — extrapolated from a 4-case smoke whose escalation
                        rate did not represent the real pool.

None of that was necessary. The run knew its own position the entire time.

THE CONTRACT. A long run writes a small JSON file, atomically, every N items:

    {"done": 337, "total": 464, "started_utc": ..., "updated_utc": ...,
     "rate_per_min": 3.5, "eta_min": 36, "label": "...", "extra": {...}}

Anything asking "how far along is it?" READS THAT FILE. It does not count
cache entries, parse logs, or reconstruct position from side effects. If the
file is missing or stale, the honest answer is "unknown" — which is a far
better answer than a confident wrong one.

Two properties make it trustworthy:
  * `rate_per_min` is MEASURED over a trailing window of real completions,
    never extrapolated from a pilot;
  * `updated_utc` lets a reader detect a STALLED run — a progress file that
    stopped moving is evidence, and staleness must never read as progress.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

# A reader treats a progress file older than this as STALE rather than as
# current position. Deliberately generous: a slow LLM-paced item can take
# minutes, and crying "stalled" at a merely-slow run is its own false alarm.
STALE_AFTER_S = 300.0


class RunProgress:
    """Writes the progress file. Cheap enough to call on every item."""

    def __init__(self, path: str | Path, total: Optional[int],
                 label: str = "", window: int = 20,
                 min_interval_s: float = 2.0):
        self.path = Path(path)
        self.total = total
        self.label = label
        self.done = 0
        self._started = time.time()
        self._last_write = 0.0
        self._min_interval = min_interval_s
        # Trailing window of completion timestamps. A rate computed over the
        # WHOLE run would be dragged down by a slow start (or by a replayed
        # cache prefix that completed in milliseconds) and would not reflect
        # what the next item will cost.
        self._marks: deque = deque(maxlen=max(2, window))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rate_per_min(self) -> Optional[float]:
        if len(self._marks) < 2:
            return None
        span = self._marks[-1] - self._marks[0]
        if span <= 0:
            return None
        return (len(self._marks) - 1) / span * 60.0

    def tick(self, n: int = 1, extra: Optional[Dict[str, Any]] = None,
             force: bool = False) -> None:
        self.done += n
        self._marks.append(time.time())
        now = time.time()
        if not force and (now - self._last_write) < self._min_interval:
            return
        self._write(extra)

    def _write(self, extra: Optional[Dict[str, Any]] = None) -> None:
        r = self.rate_per_min()
        remaining = (self.total - self.done) if self.total else None
        blob = {
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "pct": (round(100.0 * self.done / self.total, 1)
                    if self.total else None),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(self._started)),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
            "elapsed_min": round((time.time() - self._started) / 60.0, 1),
            # MEASURED over a trailing window, never extrapolated from a pilot.
            "rate_per_min": round(r, 2) if r else None,
            "eta_min": (round(remaining / r) if (r and remaining is not None
                                                 and r > 0) else None),
            "pid": os.getpid(),
            "extra": extra or {},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=1))
        os.replace(tmp, self.path)      # atomic: a reader never sees a partial
        self._last_write = time.time()

    def finish(self, note: str = "") -> None:
        self._write({"finished": True, "note": note})

    def __enter__(self):
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._write({"finished": exc_type is None,
                     "error": f"{exc_type.__name__}: {exc}" if exc else ""})
        return False


def read_progress(path: str | Path) -> Dict[str, Any]:
    """Read a progress file and say plainly what is known.

    Returns a dict with a `status` of:
      running / finished / STALLED / missing / unreadable

    STALLED is a first-class outcome, not an absence. A progress file that
    stopped updating looks identical to a healthy one if you only read
    `done`, and "it hasn't moved" is exactly the fact a reader needs.
    """
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": str(p),
                "note": "no progress file — position is UNKNOWN, not zero"}
    try:
        blob = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "path": str(p),
                "note": f"{type(exc).__name__}: {exc}"}
    age = time.time() - p.stat().st_mtime
    blob["age_s"] = round(age)
    if (blob.get("extra") or {}).get("finished"):
        blob["status"] = "finished"
    elif age > STALE_AFTER_S:
        blob["status"] = "STALLED"
        blob["note"] = (f"progress file has not moved in {age:.0f}s — the run "
                        f"may be wedged; do NOT read `done` as current")
    else:
        blob["status"] = "running"
    return blob


def render(path: str | Path) -> str:
    """One-line human summary. Says UNKNOWN rather than guessing."""
    b = read_progress(path)
    st = b.get("status")
    if st in ("missing", "unreadable"):
        return f"[{st}] {b.get('note', '')}"
    head = f"[{st}] {b.get('label') or 'run'}: {b.get('done')}"
    if b.get("total"):
        head += f"/{b['total']} ({b.get('pct')}%)"
    else:
        head += "/? (total unknown)"
    if b.get("rate_per_min"):
        head += f"  {b['rate_per_min']}/min"
        if b.get("eta_min") is not None:
            head += f"  ETA ~{b['eta_min']}min (measured)"
    else:
        head += "  rate unknown"
    if st == "STALLED":
        head += f"  ⚠ {b.get('note')}"
    return head
