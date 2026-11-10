"""§4SX Phase C: the host-side run registry (UNIFIED_DESIGN §3.2, docs/reference/labs/PHASE_C.md).

One row per thing that keeps executing after a tool call returns, wherever it
runs. It lives under ``<memory_dir>`` — NEVER on the sandbox bind mount, which
the supervised commands can write (P6).

The row is a POINTER, not the truth: a workspace run IS a services/jobs
supervisor row (``backend`` + ``backend_ref``), a lab run IS a labs-host job.
State is PROBED by the caller (``tools/runs.py``) and written back here with
``set_state``; this module never decides a run's state on its own.

Small and synchronous; every method swallows its own errors and logs (the tool
must still answer — and say so — with the registry unavailable).
"""
from __future__ import annotations

import contextlib
import json as _json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    place         TEXT NOT NULL,
    lifetime      TEXT NOT NULL,
    command       TEXT NOT NULL DEFAULT '',
    port          INTEGER,
    owner         TEXT NOT NULL DEFAULT '',
    origin        TEXT NOT NULL DEFAULT '',
    backend       TEXT NOT NULL,
    backend_ref   TEXT NOT NULL DEFAULT '',
    backend_token TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    state         TEXT NOT NULL DEFAULT 'running',
    state_note    TEXT NOT NULL DEFAULT '',
    last_probe_at REAL,
    ended_at      REAL,
    meta          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS runs_owner ON runs(owner);
"""

#: states after which a row is history (its name may be reused)
ENDED = frozenset({"stopped", "exited", "failed", "lost", "replaced", "expired"})
#: ended rows older than this are pruned on write
PRUNE_AFTER_S = 30 * 86400


class RunRegistry:
    def __init__(self, memory_dir):
        self.path = Path(memory_dir) / "runs.db"
        self._lock = threading.Lock()
        with contextlib.closing(self._connect()) as c, c:
            c.executescript(_SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
            if "meta" not in cols:                       # §4SX D3: rows from before the column
                c.execute("ALTER TABLE runs ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'")

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def add(self, *, name: str, place: str, lifetime: str, command: str, port: Optional[int],
            owner: str, origin: str, backend: str, backend_ref: str, backend_token: str = "",
            meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """A new live row; returns its run_id (None when not recorded). ``meta`` is written in the SAME insert
        (§4SX D5 review: a leaf tag set after the insert was lost when the leaf was cancelled in between)."""
        try:
            rid = "run_" + uuid.uuid4().hex[:8]
            now = time.time()
            _m = _json.dumps({k: v for k, v in (meta or {}).items() if v is not None}, sort_keys=True)
            with self._lock, contextlib.closing(self._connect()) as c, c:
                c.execute("INSERT INTO runs (run_id, name, place, lifetime, command, port, owner, origin, backend, "
                          "backend_ref, backend_token, started_at, state, last_probe_at, meta) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (rid, name, place, lifetime, (command or "")[:2000], port, owner or "", origin or "",
                           backend, backend_ref or "", backend_token or "", now, "running", now, _m))
                c.execute("DELETE FROM runs WHERE ended_at IS NOT NULL AND ended_at < ?", (now - PRUNE_AFTER_S,))
            return rid
        except Exception:  # noqa: BLE001
            logger.warning("run registry: start not recorded", exc_info=True)
            return None

    def set_state(self, run_id: str, state: str, note: str = "", *, only_live: bool = False,
                  revive: bool = False) -> None:
        """Write a PROBED state back. An ended state stamps ``ended_at`` once. ``only_live``: never over a row
        that already ended (review r2: a landed-job write-back racing a `run stop` turned "stopped" into "exited").
        A LIVE state never lands on an ended row (review 11-05 R-m8: a probe that read before a concurrent stop wrote
        "running" over "stopped", and the row was live by state, ended by timestamp) — except ``revive``: the caller
        KNOWS the process runs again (a restart that could not stop it), which clears ``ended_at``."""
        try:
            now = time.time()
            with self._lock, contextlib.closing(self._connect()) as c, c:
                if state in ENDED:
                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=?, "
                              "ended_at=COALESCE(ended_at, ?) WHERE run_id=?"
                              + (" AND ended_at IS NULL" if only_live else ""),
                              (state, (note or "")[:300], now, now, run_id))
                elif revive:
                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=?, ended_at=NULL WHERE run_id=?",
                              (state, (note or "")[:300], now, run_id))
                else:
                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=? WHERE run_id=? "
                              "AND ended_at IS NULL", (state, (note or "")[:300], now, run_id))
        except Exception:  # noqa: BLE001
            logger.warning("run registry: state not recorded", exc_info=True)

    def set_meta(self, run_id: str, **fields) -> bool:
        """Merge ``fields`` into the row's ``meta`` JSON (None removes a key). §4SX D3: the down-notice period of an
        `app` run lives here — never in ``state_note``, which every probe overwrites."""
        import json as _json
        try:
            with self._lock, contextlib.closing(self._connect()) as c, c:
                r = c.execute("SELECT meta FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if r is None:
                    return False
                try:
                    m = _json.loads(r["meta"] or "{}")
                except ValueError:
                    m = {}
                for k, v in fields.items():
                    if v is None:
                        m.pop(k, None)
                    else:
                        m[k] = v
                c.execute("UPDATE runs SET meta=? WHERE run_id=?", (_json.dumps(m, sort_keys=True), run_id))
            return True
        except Exception:  # noqa: BLE001
            logger.warning("run registry: meta not recorded", exc_info=True)
            return False

    @staticmethod
    def meta_of(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        import json as _json
        try:
            return dict(_json.loads((row or {}).get("meta") or "{}"))
        except (TypeError, ValueError):
            return {}

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock, contextlib.closing(self._connect()) as c, c:
                r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                return dict(r) if r else None
        except Exception:  # noqa: BLE001
            return None

    def live_by_name(self, owner: str, name: str) -> Optional[Dict[str, Any]]:
        """The owner's newest row of that name that is not ended."""
        for r in reversed(self.rows(owner)):
            if r["name"] == name and r["state"] not in ENDED:
                return r
        return None

    def latest_by_name(self, owner: str, name: str) -> Optional[Dict[str, Any]]:
        """The owner's newest row of that name, ended or not (collect/logs of a finished job)."""
        rows = [r for r in self.rows(owner) if r["name"] == name]
        return rows[-1] if rows else None

    def rows(self, owner: Optional[str] = None, *, live_only: bool = False,
             strict: bool = False) -> List[Dict[str, Any]]:
        """``strict`` raises a read error instead of answering ``[]`` (a teardown must not read "nothing")."""
        try:
            q, a, where = "SELECT * FROM runs", [], []
            if owner is not None:
                where.append("owner = ?")
                a.append(owner)
            if live_only:
                where.append("ended_at IS NULL")
            if where:
                q += " WHERE " + " AND ".join(where)
            with self._lock, contextlib.closing(self._connect()) as c, c:
                return [dict(r) for r in c.execute(q + " ORDER BY started_at, rowid", a).fetchall()]
        except Exception:  # noqa: BLE001
            if strict:
                raise
            return []


_INSTANCES: Dict[str, RunRegistry] = {}
_INST_LOCK = threading.Lock()


def registry_for(context) -> Optional[RunRegistry]:
    """The registry under the context's memory dir (one per dir), or None."""
    md = getattr(context, "memory_dir", None)
    if not md:
        return None
    key = str(md)
    with _INST_LOCK:
        reg = _INSTANCES.get(key)
        if reg is None:
            try:
                reg = _INSTANCES[key] = RunRegistry(md)
            except Exception:  # noqa: BLE001
                logger.warning("run registry unavailable", exc_info=True)
                return None
        return reg
