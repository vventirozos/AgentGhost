"""§4SX Phase B: the agent's own record of the machines (labs) it made.

The labs host is the source of truth for what is RUNNING; this table is the
agent's memory of what it CREATED, for whom, on which host:

* ``owner`` — ``project:<id>`` or ``conversation:<key>`` (UNIFIED §3.4), so a
  conversation's "destroy the lab you just made" resolves to its own labs and
  a later phase can tear a project's labs down with the project;
* ``host_id`` — the labs host's identity at create time (UNIFIED §11): a row
  whose host changed is shown as ``missing (host changed)``, never silently
  dropped;
* ``destroyed_at`` / ``destroyed_by`` — a lab destroyed by ANOTHER owner key is
  named in the owner's next listing (UNIFIED §12 finding 13);
* ``expires_hard`` — when the lab is gone AT THE LATEST (the host's maximum
  lifetime, written at create; NULL = unknown, or a kept machine). §4UD F10:
  a reader with no host call (the briefing's RUNNING block) marks a row past
  it instead of listing a lab that cannot exist any more. It never closes the
  row: only the host's answer does that.

Stored references are ids, never URLs (UNIFIED §11: every URL comes from the
labs API at read time). Small and synchronous; every method swallows its own
errors (the tool must work with the inventory unavailable).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labs (
    lab_id       TEXT PRIMARY KEY,
    owner        TEXT NOT NULL DEFAULT '',
    origin       TEXT NOT NULL DEFAULT '',
    template     TEXT NOT NULL DEFAULT '',
    label        TEXT NOT NULL DEFAULT '',
    purpose      TEXT NOT NULL DEFAULT '',
    host_id      TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    destroyed_at REAL,
    destroyed_by TEXT NOT NULL DEFAULT '',
    leftovers    TEXT NOT NULL DEFAULT '',
    leaf         TEXT NOT NULL DEFAULT '',
    expires_hard REAL
);
CREATE INDEX IF NOT EXISTS labs_owner ON labs(owner);
"""

#: rows destroyed this long ago are pruned on write
PRUNE_AFTER_S = 30 * 86400


class LabInventory:
    def __init__(self, memory_dir):
        self.path = Path(memory_dir) / "lab_inventory.db"
        self._lock = threading.Lock()
        with self._connect() as c:
            c.executescript(_SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(labs)").fetchall()}
            if "leftovers" not in cols:                 # §4SX D2 r5: rows from before the column
                c.execute("ALTER TABLE labs ADD COLUMN leftovers TEXT NOT NULL DEFAULT ''")
            if "leaf" not in cols:                      # §4SX D4 r2 #5: the autopilot leaf that made the lab
                c.execute("ALTER TABLE labs ADD COLUMN leaf TEXT NOT NULL DEFAULT ''")
            if "expires_hard" not in cols:              # §4UD F10: when the lab is gone at the latest
                c.execute("ALTER TABLE labs ADD COLUMN expires_hard REAL")

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def record_create(self, lab_id: str, *, owner: str, origin: str, template: str,
                      label: str = "", purpose: str = "", host_id: str = "", leaf: str = "",
                      expires_hard: Optional[float] = None) -> None:
        try:
            now = time.time()
            with self._lock, self._connect() as c:
                c.execute("INSERT OR REPLACE INTO labs (lab_id, owner, origin, template, label, purpose, host_id, "
                          "created_at, leaf, expires_hard) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (lab_id, owner or "", origin or "", template or "", (label or "")[:80],
                           (purpose or "")[:120], host_id or "", now, (leaf or "")[:80],
                           float(expires_hard) if expires_hard else None))
                c.execute("DELETE FROM labs WHERE destroyed_at IS NOT NULL AND destroyed_at < ?",
                          (now - PRUNE_AFTER_S,))
        except Exception:  # noqa: BLE001
            logger.warning("lab inventory: create not recorded", exc_info=True)

    def record_destroy(self, lab_id: str, *, by: str) -> None:
        try:
            with self._lock, self._connect() as c:
                c.execute("UPDATE labs SET destroyed_at = ?, destroyed_by = ? WHERE lab_id = ? "
                          "AND destroyed_at IS NULL", (time.time(), by or "", lab_id))
        except Exception:  # noqa: BLE001
            logger.warning("lab inventory: destroy not recorded", exc_info=True)

    def set_expiry(self, lab_id: str, expires_hard: Optional[float]) -> None:
        """§4UD F10: the hard expiry of a live row; None clears it (`lab keep`: a kept machine has none)."""
        try:
            with self._lock, self._connect() as c:
                c.execute("UPDATE labs SET expires_hard = ? WHERE lab_id = ? AND destroyed_at IS NULL",
                          (float(expires_hard) if expires_hard else None, lab_id))
        except Exception:  # noqa: BLE001
            logger.warning("lab inventory: expiry not recorded", exc_info=True)

    def clear_leaf(self, lab_id: str) -> None:
        """§4UD r2: the lab is no longer its leaf's (the owner took it over): no leaf end, sweep or boot pass
        acts on it again. The row itself stays live."""
        try:
            with self._lock, self._connect() as c:
                c.execute("UPDATE labs SET leaf = '' WHERE lab_id = ?", (lab_id,))
        except Exception:  # noqa: BLE001
            logger.warning("lab inventory: leaf tag not cleared", exc_info=True)

    def mark_leftovers(self, lab_id: str, leftovers: str) -> None:
        """§4SX D2 r5: the host destroyed the lab but could not remove everything. The row stays LIVE with the
        leftovers named (a `lab list` must not retire it as expired; a project delete must still see it)."""
        try:
            with self._lock, self._connect() as c:
                c.execute("UPDATE labs SET leftovers = ? WHERE lab_id = ? AND destroyed_at IS NULL",
                          ((leftovers or "")[:300], lab_id))
        except Exception:  # noqa: BLE001
            logger.warning("lab inventory: leftovers not recorded", exc_info=True)

    def get(self, lab_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock, self._connect() as c:
                r = c.execute("SELECT * FROM labs WHERE lab_id = ?", (lab_id,)).fetchone()
                return dict(r) if r else None
        except Exception:  # noqa: BLE001
            return None

    def rows(self, owner: Optional[str] = None, *, live_only: bool = False,
             strict: bool = False) -> List[Dict[str, Any]]:
        """``strict`` raises a read error instead of answering ``[]`` (a teardown must not read "nothing")."""
        try:
            q, a = "SELECT * FROM labs", []
            where = []
            if owner is not None:
                where.append("owner = ?")
                a.append(owner)
            if live_only:
                where.append("destroyed_at IS NULL")
            if where:
                q += " WHERE " + " AND ".join(where)
            with self._lock, self._connect() as c:
                return [dict(r) for r in c.execute(q + " ORDER BY created_at", a).fetchall()]
        except Exception:  # noqa: BLE001
            if strict:
                raise
            return []


_INSTANCES: Dict[str, LabInventory] = {}
_INST_LOCK = threading.Lock()


def inventory_for(context) -> Optional[LabInventory]:
    """The inventory under the context's memory dir (one per dir), or None."""
    md = getattr(context, "memory_dir", None)
    if not md:
        return None
    key = str(md)
    with _INST_LOCK:
        inv = _INSTANCES.get(key)
        if inv is None:
            try:
                inv = _INSTANCES[key] = LabInventory(md)
            except Exception:  # noqa: BLE001
                logger.warning("lab inventory unavailable", exc_info=True)
                return None
        return inv
