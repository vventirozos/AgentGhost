import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Optional
from collections import OrderedDict

logger = logging.getLogger("GhostAgent")

# Default TTL: 24 hours (in seconds)
_DEFAULT_TTL = 86400

# Cap on the value echoed back by `set()`. The return string is handed
# straight to the model (tools/memory.py `remember`), so echoing a multi-KB
# swarm result doubled it into the context for zero information gain — the
# model just wrote it. The stored value is NEVER truncated (that would
# corrupt the very results this scratchpad exists to protect); only the
# acknowledgement is bounded.
_DEFAULT_MAX_ECHO_CHARS = 500

# Sentinel for "no namespace filter given". `None` is a REAL namespace (the
# global/free-chat scope), so it can't double as "unset".
_UNSET = object()


class Scratchpad:
    """In-memory LRU scratchpad with optional SQLite persistence.

    When ``persist_path`` is provided, entries survive process restarts.
    A TTL (default 24 hours) auto-expires stale entries on load. When no
    persist path is given, the scratchpad is purely in-memory (original
    behaviour, used by self-play isolation contexts).

    Scopes (2026-07-22)
    -------------------
    One process-wide scratchpad is shared by every conversation, project and
    background job, so "clear the previous project's keys" used to mean
    "delete everything that isn't a sentinel" — which silently destroyed
    in-flight swarm results owned by another conversation (verified in prod:
    the live DB was down to its 2 sentinel rows).

    Each entry now carries a NAMESPACE tag (``None`` == global/free-chat).
    ``set()`` tags with ``active_namespace`` unless told otherwise, and
    :meth:`clear_namespace` deletes ONE scope, so a project switch can drop
    exactly the outgoing project's keys and nothing else.

    The tag is metadata, NOT part of the key: the key a caller writes is the
    key everyone else reads. Folding the namespace into the key string would
    have broken every cross-subsystem lookup that addresses a key by its bare
    name (``tools/swarm.py`` ``output_key``, the job registry's
    ``result_resolver``, ``recall``, the ``proj::``/sentinel conventions in
    ``tools/projects.py``) — i.e. it would have re-created the same
    "collect returns None" failure from the other side.
    """

    def __init__(self, max_entries: int = 50, persist_path: Path = None, ttl: int = _DEFAULT_TTL,
                 max_echo_chars: int = _DEFAULT_MAX_ECHO_CHARS):
        self._data: OrderedDict = OrderedDict()
        # Per-entry insertion timestamps used to enforce TTL on `get()`.
        # Without this the in-memory cache served stale entries indefinitely
        # — `_load_from_db` only purged at construction time.
        self._timestamps: dict = {}
        # key -> namespace tag (None == global scope, i.e. owned by nobody
        # in particular and never dropped by a scope clear).
        self._scopes: dict = {}
        # Namespace applied to writes that don't name one. Set by the project
        # layer when a project is activated/parked; None == free chat.
        self.active_namespace: Optional[str] = None
        self.max_entries = max_entries
        self.max_echo_chars = max_echo_chars
        self.persist_path = persist_path
        self.ttl = ttl
        self._lock = threading.RLock()
        if self.persist_path:
            self._init_db()
            self._load_from_db()

    def _init_db(self):
        # A corrupt/unreadable DB must not take down boot (the launchd
        # supervisor would respawn-loop) — degrade to in-memory instead.
        try:
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS scratchpad (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        created_at REAL,
                        accessed_at REAL,
                        namespace TEXT
                    )
                ''')
                # Migrate pre-scope DBs in place: without the column the
                # scope tags would be lost on every restart and project
                # isolation would silently stop working.
                cols = {row[1] for row in conn.execute("PRAGMA table_info(scratchpad)")}
                if "namespace" not in cols:
                    conn.execute("ALTER TABLE scratchpad ADD COLUMN namespace TEXT")
                conn.commit()
        except Exception as e:
            logger.warning(
                f"Scratchpad DB unusable ({self.persist_path}): {e} — "
                f"falling back to in-memory (state will not survive restarts)"
            )
            self.persist_path = None

    def _load_from_db(self):
        """Load non-expired entries from SQLite into memory."""
        if not self.persist_path:
            return
        cutoff = time.time() - self.ttl
        try:
            with closing(sqlite3.connect(self.persist_path)) as conn:
                # Purge expired entries
                conn.execute("DELETE FROM scratchpad WHERE accessed_at < ?", (cutoff,))
                conn.commit()
                cursor = conn.execute(
                    "SELECT key, value, accessed_at, namespace FROM scratchpad "
                    "ORDER BY accessed_at ASC"
                )
                for key, value_json, accessed_at, namespace in cursor:
                    try:
                        self._data[key] = json.loads(value_json)
                    except Exception:
                        self._data[key] = value_json
                    # Seed timestamps from the persisted accessed_at so the
                    # TTL check in get() can fire on entries loaded from disk.
                    try:
                        self._timestamps[key] = float(accessed_at)
                    except (TypeError, ValueError):
                        self._timestamps[key] = time.time()
                    self._scopes[key] = namespace or None
                # Trim to max
                while len(self._data) > self.max_entries:
                    evicted, _ = self._data.popitem(last=False)
                    self._timestamps.pop(evicted, None)
                    self._scopes.pop(evicted, None)
        except Exception as e:
            logger.debug(f"Scratchpad DB load failed (non-critical): {e}")

    def _persist_entry(self, key: str, value: Any, namespace: Optional[str] = None) -> bool:
        if not self.persist_path:
            return False
        try:
            now = time.time()
            value_json = json.dumps(value, default=str)
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO scratchpad (key, value, created_at, accessed_at, namespace) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, value_json, now, now, namespace)
                )
                conn.commit()
            return True
        except Exception as e:
            # WARNING, not debug: a full disk or a locked DB makes every
            # swarm result memory-only while `set()` keeps returning
            # "Stored:" — the loss only surfaced after the next restart.
            # The operator watches the live stream, so this must be visible
            # there the moment it starts happening.
            logger.warning(
                f"Scratchpad persist FAILED for key '{key}' ({type(e).__name__}: {e}) — "
                f"the value is in memory only and will NOT survive a restart"
            )
            return False

    def _persist_access(self, key: str):
        if not self.persist_path:
            return
        try:
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute(
                    "UPDATE scratchpad SET accessed_at = ? WHERE key = ?",
                    (time.time(), key)
                )
                conn.commit()
        except Exception:
            pass

    def _persist_delete(self, key: str):
        if not self.persist_path:
            return
        try:
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute("DELETE FROM scratchpad WHERE key = ?", (key,))
                conn.commit()
        except Exception:
            pass

    def _persist_clear(self):
        if not self.persist_path:
            return
        try:
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute("DELETE FROM scratchpad")
                conn.commit()
        except Exception:
            pass

    def _echo(self, value: Any) -> str:
        """Bounded rendering of a stored value for the `set()` ack."""
        text = str(value)
        cap = self.max_echo_chars
        if cap and cap > 0 and len(text) > cap:
            return (f"{text[:cap]}… [+{len(text) - cap} chars truncated in this "
                    f"acknowledgement only — the FULL value is stored]")
        return text

    def set(self, key: str, value: Any, namespace: Any = _UNSET):
        """Store ``value`` under ``key``.

        ``namespace`` tags the entry with a scope; omitted, it inherits
        ``active_namespace``. Pass ``namespace=None`` explicitly for an entry
        that must belong to NO scope (sentinels, background-job output) and
        therefore survive every project switch.
        """
        ns = self.active_namespace if namespace is _UNSET else namespace
        ns = str(ns) if ns else None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            self._timestamps[key] = time.time()
            self._scopes[key] = ns

            # Evict oldest if over capacity
            if len(self._data) > self.max_entries:
                evicted_key, _ = self._data.popitem(last=False)
                self._timestamps.pop(evicted_key, None)
                self._scopes.pop(evicted_key, None)
                self._persist_delete(evicted_key)

            self._persist_entry(key, value, ns)
        return f"Stored: {key} = {self._echo(value)}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                # TTL check (was missing — `_load_from_db` purged at init
                # but live entries past their TTL stayed in cache forever).
                ts = self._timestamps.get(key)
                if ts is not None and (time.time() - ts) > self.ttl:
                    del self._data[key]
                    self._timestamps.pop(key, None)
                    self._scopes.pop(key, None)
                    self._persist_delete(key)
                    return None
                self._data.move_to_end(key)
                self._persist_access(key)
                # Refresh timestamp on access so the TTL behaves like a
                # sliding window (matches the persisted `accessed_at`).
                self._timestamps[key] = time.time()
                return self._data[key]
        return None

    def list_all(self, namespace: Any = _UNSET) -> str:
        """Render the scratchpad. With no argument this is every entry
        (unchanged); pass ``namespace`` to render one scope only
        (``namespace=None`` → the global/free-chat scope)."""
        with self._lock:
            if namespace is _UNSET:
                items = list(self._data.items())
            else:
                ns = str(namespace) if namespace else None
                items = [(k, v) for k, v in self._data.items()
                         if self._scopes.get(k) == ns]
            if not items:
                return "Scratchpad is empty."
            return "\n".join([f"{k}: {v}" for k, v in items])

    def clear(self):
        with self._lock:
            self._data.clear()
            self._timestamps.clear()
            self._scopes.clear()
            self._persist_clear()
        return "Scratchpad cleared."

    def delete(self, key: str) -> bool:
        """Remove a single entry. Returns True if the key existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._timestamps.pop(key, None)
                self._scopes.pop(key, None)
                self._persist_delete(key)
                return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    # ------------------------------------------------------------------ scopes

    def keys(self) -> list:
        """All live keys, oldest-touched first."""
        with self._lock:
            return list(self._data.keys())

    def namespace_of(self, key: str) -> Optional[str]:
        """Scope tag of ``key`` (None == global, or key unknown)."""
        with self._lock:
            return self._scopes.get(key)

    def namespaces(self) -> list:
        """Distinct NON-global scope tags currently in use."""
        with self._lock:
            seen = []
            for k in self._data:
                ns = self._scopes.get(k)
                if ns and ns not in seen:
                    seen.append(ns)
            return seen

    def keys_in_namespace(self, namespace: Optional[str]) -> list:
        """Keys tagged with ``namespace`` (None → the global scope)."""
        ns = str(namespace) if namespace else None
        with self._lock:
            return [k for k in self._data if self._scopes.get(k) == ns]

    def clear_namespace(self, namespace: Optional[str], protect=None) -> list:
        """Delete every entry tagged with ``namespace``; return the keys hit.

        ``protect`` is an iterable of keys to spare even inside the scope —
        used for entries a background job still owns (its result may land
        long after the scope that dispatched it was parked). Entries in OTHER
        scopes, and global (untagged) entries, are never touched.
        """
        ns = str(namespace) if namespace else None
        spared = set(protect or ())
        with self._lock:
            victims = [k for k in list(self._data.keys())
                       if self._scopes.get(k) == ns and k not in spared]
            for k in victims:
                del self._data[k]
                self._timestamps.pop(k, None)
                self._scopes.pop(k, None)
                self._persist_delete(k)
        return victims