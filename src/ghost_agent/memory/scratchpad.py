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


class Scratchpad:
    """In-memory LRU scratchpad with optional SQLite persistence.

    When ``persist_path`` is provided, entries survive process restarts.
    A TTL (default 24 hours) auto-expires stale entries on load. When no
    persist path is given, the scratchpad is purely in-memory (original
    behaviour, used by self-play isolation contexts).
    """

    def __init__(self, max_entries: int = 50, persist_path: Path = None, ttl: int = _DEFAULT_TTL):
        self._data: OrderedDict = OrderedDict()
        # Per-entry insertion timestamps used to enforce TTL on `get()`.
        # Without this the in-memory cache served stale entries indefinitely
        # — `_load_from_db` only purged at construction time.
        self._timestamps: dict = {}
        self.max_entries = max_entries
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
                        accessed_at REAL
                    )
                ''')
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
                    "SELECT key, value, accessed_at FROM scratchpad ORDER BY accessed_at ASC"
                )
                for key, value_json, accessed_at in cursor:
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
                # Trim to max
                while len(self._data) > self.max_entries:
                    evicted, _ = self._data.popitem(last=False)
                    self._timestamps.pop(evicted, None)
        except Exception as e:
            logger.debug(f"Scratchpad DB load failed (non-critical): {e}")

    def _persist_entry(self, key: str, value: Any):
        if not self.persist_path:
            return
        try:
            now = time.time()
            value_json = json.dumps(value, default=str)
            with closing(sqlite3.connect(self.persist_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO scratchpad (key, value, created_at, accessed_at) VALUES (?, ?, ?, ?)",
                    (key, value_json, now, now)
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Scratchpad persist failed (non-critical): {e}")

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

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            self._timestamps[key] = time.time()

            # Evict oldest if over capacity
            if len(self._data) > self.max_entries:
                evicted_key, _ = self._data.popitem(last=False)
                self._timestamps.pop(evicted_key, None)
                self._persist_delete(evicted_key)

            self._persist_entry(key, value)
        return f"Stored: {key} = {value}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                # TTL check (was missing — `_load_from_db` purged at init
                # but live entries past their TTL stayed in cache forever).
                ts = self._timestamps.get(key)
                if ts is not None and (time.time() - ts) > self.ttl:
                    del self._data[key]
                    self._timestamps.pop(key, None)
                    self._persist_delete(key)
                    return None
                self._data.move_to_end(key)
                self._persist_access(key)
                # Refresh timestamp on access so the TTL behaves like a
                # sliding window (matches the persisted `accessed_at`).
                self._timestamps[key] = time.time()
                return self._data[key]
        return None

    def list_all(self) -> str:
        with self._lock:
            if not self._data:
                return "Scratchpad is empty."
            return "\n".join([f"{k}: {v}" for k, v in self._data.items()])

    def clear(self):
        with self._lock:
            self._data.clear()
            self._timestamps.clear()
            self._persist_clear()
        return "Scratchpad cleared."

    def delete(self, key: str) -> bool:
        """Remove a single entry. Returns True if the key existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._timestamps.pop(key, None)
                self._persist_delete(key)
                return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._data)