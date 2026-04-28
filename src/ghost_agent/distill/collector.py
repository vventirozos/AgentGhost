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


# Filename for the per-root corrections sidecar. The sidecar records
# outcome mutations applied AFTER a trajectory was originally
# appended, e.g. when the user's next message reveals the prior turn
# was a failure. Keeping it as a separate append-only file (rather
# than rewriting the original JSONL line) preserves the audit trail
# and avoids the byte-offset bookkeeping a true mutation would
# require. Readers overlay it on top of the source JSONL when
# iterating.
CORRECTIONS_FILENAME = "corrections.jsonl"


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
    # Outcome corrections (sidecar)
    # -----------------------------------------------------------------

    def _corrections_path(self) -> Path:
        """Single file under ``self.root`` (NOT day-partitioned) so
        readers can apply corrections without scanning every day's
        directory. The corrections workload is tiny — at most one
        record per failed turn — so a single growing file is fine."""
        return self.root / CORRECTIONS_FILENAME

    def update_outcome(
        self,
        trajectory_id: str,
        new_outcome: str,
        reason: str = "",
        *,
        source: str = "",
    ) -> bool:
        """Record an outcome mutation for ``trajectory_id``.

        Appends a JSON line to the corrections sidecar; the original
        JSONL trajectory line stays exactly as it was written. The
        next ``iter_trajectories`` walk overlays the latest
        correction per id.

        ``source`` is a short label for the caller (e.g.,
        ``"user_correction"``, ``"verifier"``) — useful when
        debugging which detector promoted a trajectory.

        Returns True iff the mutation was persisted. Never raises:
        a failed correction must not break the user turn.
        """
        if not self.enabled:
            return False
        if not isinstance(trajectory_id, str) or not trajectory_id:
            return False
        try:
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            record = {
                "trajectory_id": trajectory_id,
                "outcome": str(new_outcome or ""),
                "reason": str(reason or "")[:500],
                "source": str(source or "")[:100],
                "timestamp": ts,
            }
            path = self._corrections_path()
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    import json as _json
                    f.write(_json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
            return True
        except Exception as e:
            logger.warning("trajectory correction append failed: %s", e)
            return False

    def _load_corrections(self) -> dict:
        """Read the corrections sidecar into a ``{traj_id: record}``
        dict. Later records for the same id win (append-only +
        last-write-wins). Returns an empty dict when the file is
        missing or unreadable."""
        path = self._corrections_path()
        if not path.exists():
            return {}
        out: dict = {}
        try:
            import json as _json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("trajectory_id")
                    if not isinstance(tid, str) or not tid:
                        continue
                    out[tid] = rec
        except OSError as e:
            logger.warning("cannot read corrections sidecar %s: %s", path, e)
        return out

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

    def iter_trajectories(
        self,
        *,
        day: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Trajectory]:
        """Stream trajectories from disk, overlaying outcome
        corrections from the sidecar.

        Filters:
          - `day` (YYYY-MM-DD) restricts to one partition; None walks all.
          - `session_id` restricts to one session; None walks all.

        For every trajectory whose id has a sidecar correction, the
        ``outcome`` (and ``failure_reason``, when the sidecar carries
        one) on the yielded copy reflect the correction. The on-disk
        JSONL line is never modified — readers always see the latest
        verdict, but the original write stays preserved for audit.
        """
        import json

        corrections = self._load_corrections()

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
                                traj = Trajectory.from_dict(d_obj)
                            except Exception:
                                # Schema drift — skip but don't crash the walk.
                                continue
                            corr = corrections.get(traj.id)
                            if corr:
                                new_outcome = corr.get("outcome") or ""
                                if new_outcome:
                                    traj.outcome = new_outcome
                                # Preserve any pre-existing failure_reason on
                                # the original record; only fill in from the
                                # correction when the original was empty.
                                reason = corr.get("reason") or ""
                                if reason and not (traj.failure_reason or ""):
                                    traj.failure_reason = reason
                            yield traj
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
