"""sqlite metadata store. Docker labels are the source of truth for what
exists; this holds metadata only (contract v1 §4) plus the leases, secrets
and app state that v2 adds."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS labs (
  lab_id TEXT PRIMARY KEY,
  template TEXT, label TEXT, state TEXT,
  owner TEXT, origin TEXT, lineage TEXT, lifetime TEXT, purpose TEXT,
  ttl_class TEXT, ttl_idle_s INTEGER, ttl_max_s INTEGER,
  egress TEXT, subnet TEXT,
  created_at REAL, last_used_at REAL, last_owner_use_at REAL,
  retire_at REAL,
  spec_json TEXT, readiness_json TEXT, app_json TEXT, agent_api_url TEXT, lan INTEGER DEFAULT 0,
  pool TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS members (
  lab_id TEXT, name TEXT, ip TEXT, image TEXT, mem_mb INTEGER, cpus REAL,
  cap_add TEXT, ports_json TEXT, isolated INTEGER DEFAULT 0,
  PRIMARY KEY (lab_id, name)
);
CREATE TABLE IF NOT EXISTS subnets (subnet TEXT PRIMARY KEY, lab_id TEXT);
CREATE TABLE IF NOT EXISTS ports (
  port INTEGER PRIMARY KEY, lab_id TEXT, member TEXT, cport INTEGER,
  lease_key TEXT, lan INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, lab_id TEXT, member TEXT, cmd TEXT,
  started_at REAL, ended_at REAL, state TEXT, exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS job_starts (job_id TEXT PRIMARY KEY, lab_id TEXT, container_started TEXT);
CREATE TABLE IF NOT EXISTS destroyed (
  lab_id TEXT PRIMARY KEY, at REAL, reason TEXT, by TEXT, owner TEXT
);
CREATE TABLE IF NOT EXISTS secrets (
  lineage TEXT, app TEXT, name TEXT, value TEXT,
  PRIMARY KEY (lineage, app, name)
);
CREATE TABLE IF NOT EXISTS archives (
  lineage TEXT, app TEXT, ts REAL, path TEXT, requirements_json TEXT, data_seed_json TEXT
);
CREATE TABLE IF NOT EXISTS backups (
  backup_id TEXT PRIMARY KEY, lineage TEXT, app TEXT, member TEXT, ts REAL, path TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS previews (name TEXT PRIMARY KEY, url TEXT, project TEXT, at REAL);
CREATE TABLE IF NOT EXISTS anon_volumes (name TEXT PRIMARY KEY, lab_id TEXT, member TEXT, at REAL);
CREATE TABLE IF NOT EXISTS probe_claims (kind TEXT, key TEXT, lab_id TEXT, at REAL, PRIMARY KEY (kind, key));
"""
# columns added after a host already had a database (CREATE TABLE IF NOT EXISTS never adds them)
MIGRATIONS = (
    # review 11-05 #4: the pool a lab is counted in when it differs from its lifetime ("kept": the rehearsal twin)
    ("labs", "pool", "TEXT DEFAULT ''"),
)


class Store:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        for table, col, decl in MIGRATIONS:
            have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})").fetchall()}
            if col not in have:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        # secrets live here: owner-only, whatever the umask was (r1 core #3)
        for f in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            try:
                f.chmod(0o600)
            except FileNotFoundError:
                pass

    # -- generic -----------------------------------------------------------
    def q(self, sql: str, args: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def x(self, sql: str, args: Iterable[Any] = ()) -> None:
        with self._lock:
            self._db.execute(sql, tuple(args))

    # -- labs --------------------------------------------------------------
    def put_lab(self, row: Dict[str, Any]) -> None:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        self.x(f"INSERT OR REPLACE INTO labs ({cols}) VALUES ({qs})", row.values())

    def lab(self, lab_id: str) -> Optional[Dict[str, Any]]:
        r = self.q("SELECT * FROM labs WHERE lab_id=?", (lab_id,))
        return r[0] if r else None

    def labs(self) -> List[Dict[str, Any]]:
        return self.q("SELECT * FROM labs ORDER BY created_at")

    def update_lab(self, lab_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE labs SET {sets} WHERE lab_id=?", [*fields.values(), lab_id])

    def delete_lab(self, lab_id: str) -> None:
        with self._lock:
            for t in ("labs", "members", "jobs", "job_starts"):
                self._db.execute(f"DELETE FROM {t} WHERE lab_id=?", (lab_id,))
            self._db.execute("DELETE FROM subnets WHERE lab_id=?", (lab_id,))

    # -- members -----------------------------------------------------------
    def put_member(self, row: Dict[str, Any]) -> None:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        self.x(f"INSERT OR REPLACE INTO members ({cols}) VALUES ({qs})", row.values())

    def members(self, lab_id: str) -> List[Dict[str, Any]]:
        return self.q("SELECT * FROM members WHERE lab_id=? ORDER BY rowid", (lab_id,))

    # -- destroyed memory --------------------------------------------------
    def remember_destroyed(self, lab_id: str, reason: str, by: str, owner: str) -> None:
        self.x("INSERT OR REPLACE INTO destroyed VALUES (?,?,?,?,?)", (lab_id, time.time(), reason, by, owner))

    def destroyed(self, lab_id: str, within_s: int) -> Optional[Dict[str, Any]]:
        r = self.q("SELECT * FROM destroyed WHERE lab_id=? AND at>?", (lab_id, time.time() - within_s))
        return r[0] if r else None

    def recently_destroyed(self, within_s: int) -> List[Dict[str, Any]]:
        return self.q("SELECT * FROM destroyed WHERE at>? ORDER BY at DESC", (time.time() - within_s,))

    # -- secrets -----------------------------------------------------------
    def secrets(self, lineage: str, app: str) -> Dict[str, str]:
        return {r["name"]: r["value"] for r in self.q("SELECT name,value FROM secrets WHERE lineage=? AND app=?", (lineage, app))}

    def all_secret_values(self) -> List[str]:
        return [r["value"] for r in self.q("SELECT value FROM secrets")]

    def set_secret(self, lineage: str, app: str, name: str, value: str) -> None:
        self.x("INSERT OR REPLACE INTO secrets VALUES (?,?,?,?)", (lineage, app, name, value))


def dumps(v: Any) -> str:
    return json.dumps(v, sort_keys=True)


def loads(s: Optional[str], default: Any = None) -> Any:
    if not s:
        return default
    return json.loads(s)
