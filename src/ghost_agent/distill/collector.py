"""JSONL trajectory writer.

Writes one Trajectory per line to a day-partitioned file under
`$GHOST_HOME/trajectories/YYYY-MM-DD/session-<sid>.jsonl`. Append-only,
thread-safe, crash-safe (uses line-buffered writes with explicit flush).

Designed to live inside the agent's hot path without becoming one —
every append is:
  - redaction (pure function, microseconds)
  - one file open/append/close
  - one flush

If the write fails (disk full, permission), the collector logs and
continues; a failed trajectory write must never break a user request.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .redact import redact_trajectory, RedactionConfig
from .schema import Trajectory

logger = logging.getLogger("GhostDistill")


def _default_root() -> Path:
    base = os.getenv("GHOST_HOME")
    if base:
        return Path(base) / "trajectories"
    return Path.home() / ".ghost" / "trajectories"


class TrajectoryCollector:
    """Append-only JSONL writer with day partitioning.

    Usage:
        collector = TrajectoryCollector()           # uses $GHOST_HOME
        collector.append(trajectory)
        # Later:
        for traj in collector.iter_trajectories():
            ...
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        session_id: Optional[str] = None,
        redaction: Optional[RedactionConfig] = None,
        enabled: bool = True,
    ):
        self.root = Path(root) if root is not None else _default_root()
        self.session_id = session_id or _default_session_id()
        self.redaction = redaction or RedactionConfig()
        self.enabled = enabled
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Write path
    # -----------------------------------------------------------------

    def _file_for(self, ts: datetime.datetime) -> Path:
        day = ts.strftime("%Y-%m-%d")
        return self.root / day / f"session-{self.session_id}.jsonl"

    def append(self, traj: Trajectory) -> Optional[Path]:
        """Redact and append `traj`. Returns the path written to, or
        None if disabled / failed.

        Never raises: trajectory logging is secondary to the agent's
        primary job.
        """
        if not self.enabled:
            return None
        try:
            redacted = redact_trajectory(traj, self.redaction)
            ts = datetime.datetime.utcnow()
            path = self._file_for(ts)
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(redacted.to_jsonl())
                    f.write("\n")
                    f.flush()
            return path
        except Exception as e:
            logger.warning("trajectory append failed: %s", e)
            return None

    def append_many(self, trajs: Iterable[Trajectory]) -> int:
        """Batch append; returns the number successfully written."""
        n = 0
        for t in trajs:
            if self.append(t) is not None:
                n += 1
        return n

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

    def iter_trajectories(
        self,
        *,
        day: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Trajectory]:
        """Stream trajectories from disk.

        Filters:
          - `day` (YYYY-MM-DD) restricts to one partition; None walks all.
          - `session_id` restricts to one session; None walks all.
        """
        import json

        day_dirs: List[Path]
        if day:
            day_dirs = [self.root / day]
        else:
            if not self.root.exists():
                return
            day_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])

        for d in day_dirs:
            if not d.exists():
                continue
            for file_path in sorted(d.glob("session-*.jsonl")):
                if session_id and f"session-{session_id}.jsonl" != file_path.name:
                    continue
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d_obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            try:
                                yield Trajectory.from_dict(d_obj)
                            except Exception:
                                # Schema drift — skip but don't crash the walk.
                                continue
                except OSError as e:
                    logger.warning("cannot read trajectory file %s: %s", file_path, e)

    def count(self) -> int:
        """Cheap count: iterates lazily without parsing the whole trajectory."""
        if not self.root.exists():
            return 0
        n = 0
        for day_dir in self.root.iterdir():
            if not day_dir.is_dir():
                continue
            for file_path in day_dir.glob("session-*.jsonl"):
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                n += 1
                except OSError:
                    continue
        return n


def _default_session_id() -> str:
    """Default session ID: compact UTC date+hex fragment. Fine for
    non-conversation callers; real conversations should pass their own.
    """
    import uuid
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"
