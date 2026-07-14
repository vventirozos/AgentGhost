"""Tests for scratchpad persistence (#12).

Verifies that:
- Scratchpad persists to SQLite and survives restarts
- TTL expiration works
- LRU eviction persists correctly
- In-memory-only mode (no persist_path) still works
"""

import pytest
import sqlite3
import time
from pathlib import Path
from ghost_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def persistent_scratchpad(tmp_path):
    db_path = tmp_path / "scratchpad.db"
    return Scratchpad(max_entries=10, persist_path=db_path, ttl=86400)


@pytest.fixture
def memory_only_scratchpad():
    return Scratchpad(max_entries=10)


class TestPersistentScratchpad:
    def test_set_and_get(self, persistent_scratchpad):
        persistent_scratchpad.set("key1", "value1")
        assert persistent_scratchpad.get("key1") == "value1"

    def test_persistence_across_instances(self, tmp_path):
        db_path = tmp_path / "scratchpad.db"
        sp1 = Scratchpad(max_entries=10, persist_path=db_path)
        sp1.set("persistent_key", {"nested": "data"})

        # New instance should load from DB
        sp2 = Scratchpad(max_entries=10, persist_path=db_path)
        result = sp2.get("persistent_key")
        assert result == {"nested": "data"}

    def test_ttl_expiration(self, tmp_path):
        db_path = tmp_path / "scratchpad.db"
        # Create with very short TTL
        sp1 = Scratchpad(max_entries=10, persist_path=db_path, ttl=1)
        sp1.set("expires_soon", "data")

        # Wait for TTL
        time.sleep(1.5)

        # New instance should not load expired entry
        sp2 = Scratchpad(max_entries=10, persist_path=db_path, ttl=1)
        assert sp2.get("expires_soon") is None

    def test_lru_eviction_persists(self, tmp_path):
        db_path = tmp_path / "scratchpad.db"
        sp = Scratchpad(max_entries=3, persist_path=db_path)

        sp.set("a", 1)
        sp.set("b", 2)
        sp.set("c", 3)
        sp.set("d", 4)  # Should evict "a"

        assert sp.get("a") is None
        assert sp.get("d") == 4

        # Verify in new instance
        sp2 = Scratchpad(max_entries=3, persist_path=db_path)
        assert sp2.get("a") is None
        assert sp2.get("d") == 4

    def test_list_all(self, persistent_scratchpad):
        persistent_scratchpad.set("x", 1)
        persistent_scratchpad.set("y", 2)
        result = persistent_scratchpad.list_all()
        assert "x: 1" in result
        assert "y: 2" in result

    def test_clear_persists(self, tmp_path):
        db_path = tmp_path / "scratchpad.db"
        sp1 = Scratchpad(max_entries=10, persist_path=db_path)
        sp1.set("key", "value")
        sp1.clear()

        sp2 = Scratchpad(max_entries=10, persist_path=db_path)
        assert sp2.get("key") is None
        assert sp2.list_all() == "Scratchpad is empty."

    def test_delete_single_key(self, persistent_scratchpad):
        persistent_scratchpad.set("k1", "v1")
        persistent_scratchpad.set("k2", "v2")
        assert persistent_scratchpad.delete("k1") is True
        assert persistent_scratchpad.get("k1") is None
        assert persistent_scratchpad.get("k2") == "v2"

    def test_delete_nonexistent_key(self, persistent_scratchpad):
        assert persistent_scratchpad.delete("nonexistent") is False

    def test_count(self, persistent_scratchpad):
        assert persistent_scratchpad.count() == 0
        persistent_scratchpad.set("a", 1)
        persistent_scratchpad.set("b", 2)
        assert persistent_scratchpad.count() == 2


class TestConnectionHygiene:
    def test_connections_closed_after_operations(self, tmp_path, monkeypatch):
        """Every sqlite connection must be closed after its operation.

        `with sqlite3.connect(...)` only wraps the *transaction* — it never
        closes the connection — so each persistence helper wraps the connect
        in contextlib.closing (§4B "scratchpad connections not closed").
        """
        import ghost_agent.memory.scratchpad as sp_mod
        opened = []
        real_connect = sp_mod.sqlite3.connect

        def tracking_connect(*a, **kw):
            conn = real_connect(*a, **kw)
            opened.append(conn)
            return conn

        monkeypatch.setattr(sp_mod.sqlite3, "connect", tracking_connect)
        sp = Scratchpad(max_entries=5, persist_path=tmp_path / "scratchpad.db")
        sp.set("k", "v")
        sp.get("k")
        sp.delete("k")
        sp.clear()

        assert opened, "expected sqlite connections to be opened"
        for conn in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")


class TestCorruptDbFallback:
    def test_corrupt_db_falls_back_to_memory(self, tmp_path):
        """A corrupt scratchpad.db must not crash construction.

        The scratchpad is built during prod boot; an unguarded raise here
        would put the launchd KeepAlive supervisor into a respawn loop.
        """
        db_path = tmp_path / "scratchpad.db"
        db_path.write_bytes(b"this is not a sqlite database " * 20)

        sp = Scratchpad(max_entries=5, persist_path=db_path)  # must not raise
        assert sp.persist_path is None  # degraded to in-memory
        sp.set("k", "v")
        assert sp.get("k") == "v"


class TestMemoryOnlyScratchpad:
    def test_basic_operations(self, memory_only_scratchpad):
        sp = memory_only_scratchpad
        sp.set("key", "value")
        assert sp.get("key") == "value"
        assert sp.count() == 1
        sp.clear()
        assert sp.count() == 0

    def test_lru_eviction(self, memory_only_scratchpad):
        sp = memory_only_scratchpad
        for i in range(15):
            sp.set(f"k{i}", i)
        # Max is 10, oldest should be evicted
        assert sp.get("k0") is None
        assert sp.get("k14") == 14
        assert sp.count() == 10

    def test_empty_scratchpad(self, memory_only_scratchpad):
        assert memory_only_scratchpad.list_all() == "Scratchpad is empty."
        assert memory_only_scratchpad.get("anything") is None
