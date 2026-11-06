"""Cross-process write-safety for the two memory stores that were doing
an unguarded read-modify-write across separate lock acquisitions.

Both defects had the same root cause: an in-process ``threading.RLock``
only serialises writers INSIDE one process, but Slack, web and CLI each
run their own process against the SAME store.

1. ``memory/projects.py`` — ``append_ledger`` / ``set_config_value`` /
   ``set_ledger`` read the project metadata under the lock, released it,
   then wrote under the lock again, so a concurrent writer in another
   process could interleave and lose an update. The fix makes the whole
   read-modify-write a single ``BEGIN IMMEDIATE`` transaction (SQLite's
   write lock is cross-connection, so this serialises processes too).

2. ``memory/skills.py`` — ``_save_playbook_unlocked`` wrote to a FIXED
   ``.tmp`` path, so two processes writing at once produced a torn temp
   file that ``os.replace`` then promoted. The fix gives the temp file a
   PID-unique name AND wraps the write + replace in an fcntl advisory lock
   on a sibling ``.lock`` file (mirrors ``memory/frontier.py``), with a
   graceful no-op fallback where fcntl is unavailable.

The concurrency tests use two independent store instances (hence two
independent RLocks) driven from threads: for projects.py that exercises
the cross-CONNECTION SQLite locking the fix relies on; for skills.py the
fcntl lock contends across the two file descriptors within one process.
"""

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.projects import ProjectStore


# ======================================================================
# projects.py — atomic metadata read-modify-write
# ======================================================================

@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def test_metadata_rmw_uses_begin_immediate(tmp_path, monkeypatch):
    """append_ledger / set_config_value must drive their read-modify-write
    through a single ``BEGIN IMMEDIATE`` transaction. Regresses if a caller
    reverts to the old get_project + _write_metadata (two-lock) shape."""
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P")

    executed = []

    class _RecConn(sqlite3.Connection):
        def execute(self, sql, *a, **k):
            executed.append(sql)
            return super().execute(sql, *a, **k)

    orig_connect = sqlite3.connect

    def _patched(path, *a, **k):
        k.setdefault("factory", _RecConn)
        return orig_connect(path, *a, **k)

    monkeypatch.setattr(
        "ghost_agent.memory.projects.sqlite3.connect", _patched
    )

    executed.clear()
    store.append_ledger(pid, "entrypoint is main.py")
    assert any("BEGIN IMMEDIATE" in s.upper() for s in executed), (
        "append_ledger did not open a BEGIN IMMEDIATE transaction"
    )

    executed.clear()
    store.set_config_value(pid, "port", "8000")
    assert any("BEGIN IMMEDIATE" in s.upper() for s in executed), (
        "set_config_value did not open a BEGIN IMMEDIATE transaction"
    )


def test_atomic_metadata_update_no_lost_increment(tmp_path):
    """The classic lost-update probe: many threads, using TWO independent
    ProjectStore instances (two RLocks), each read-increment a counter in
    project metadata via the atomic primitive. Every increment must land —
    a lost read-modify-write would leave the final total below N·rounds.
    This does NOT self-heal (we count total increments), so it fails on the
    old two-lock design and passes only when the RMW is truly atomic."""
    mem = tmp_path / "mem"
    sb = tmp_path / "sb"
    store_a = ProjectStore(mem, sandbox_root=sb)
    store_b = ProjectStore(mem, sandbox_root=sb)  # separate RLock, same DB
    pid = store_a.create_project("P")

    ROUNDS = 150
    THREADS = 6

    def _incr(meta):
        meta["counter"] = int(meta.get("counter") or 0) + 1
        return meta

    def worker(store):
        for _ in range(ROUNDS):
            store._atomic_metadata_update(pid, _incr)

    threads = [
        threading.Thread(target=worker, args=(store_a if i % 2 else store_b,))
        for i in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = int((store_a.get_project(pid)["metadata"]).get("counter"))
    assert final == ROUNDS * THREADS, (
        f"lost update: final counter {final} != expected {ROUNDS * THREADS}"
    )


def test_concurrent_set_config_keeps_all_keys(tmp_path):
    """End-to-end: each thread (across two stores) owns a distinct config
    key and hammers set_config_value. The internal RMW reads the WHOLE
    config map, so a non-atomic writer could drop a sibling's key. Every
    key must survive with its last value."""
    mem = tmp_path / "mem"
    sb = tmp_path / "sb"
    store_a = ProjectStore(mem, sandbox_root=sb)
    store_b = ProjectStore(mem, sandbox_root=sb)
    pid = store_a.create_project("P")

    ROUNDS = 60
    THREADS = 6

    def worker(store, idx):
        key = f"key_{idx}"
        for r in range(ROUNDS):
            store.set_config_value(pid, key, str(r))

    threads = [
        threading.Thread(
            target=worker, args=(store_a if i % 2 else store_b, i)
        )
        for i in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cfg = store_a.get_config(pid)
    for i in range(THREADS):
        assert f"key_{i}" in cfg, f"key_{i} was clobbered by a concurrent write"
        assert cfg[f"key_{i}"] == str(ROUNDS - 1)


# ======================================================================
# skills.py — PID-unique temp + fcntl advisory lock
# ======================================================================

@pytest.fixture
def mem_dir(tmp_path):
    # SkillMemory (unlike ProjectStore) does not mkdir its memory_dir.
    d = tmp_path / "mem"
    d.mkdir()
    return d


