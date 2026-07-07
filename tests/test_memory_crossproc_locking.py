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
from ghost_agent.memory import skills as skills_mod
from ghost_agent.memory.skills import SkillMemory


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


def test_save_playbook_temp_path_is_pid_unique(mem_dir, monkeypatch):
    """The atomic save must promote a temp file whose name carries this
    process's PID, so two processes never write the SAME ``.tmp``."""
    sm = SkillMemory(mem_dir)

    captured = {}
    real_replace = os.replace

    def _spy_replace(src, dst, *a, **k):
        captured["src"] = str(src)
        captured["dst"] = str(dst)
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(skills_mod.os, "replace", _spy_replace)

    sm.save_playbook([{"task": "t", "mistake": "m", "solution": "s"}])

    src_name = Path(captured["src"]).name
    assert str(os.getpid()) in src_name, (
        f"temp path {src_name!r} does not contain the PID"
    )
    assert src_name.endswith(".tmp")
    assert Path(captured["dst"]).name == "skills_playbook.json"
    # And the save actually landed intact.
    assert sm._load_playbook()[0]["task"] == "t"


def test_crossproc_lock_is_exclusive(mem_dir):
    """Inside ``_crossproc_lock`` an exclusive fcntl lock is held on the
    sibling ``.lock`` file: a second fd cannot acquire it non-blocking, and
    can once the context exits. This is what serialises two processes /
    two SkillMemory instances through the save."""
    fcntl = pytest.importorskip("fcntl")
    sm = SkillMemory(mem_dir)

    with sm._crossproc_lock():
        assert sm._lock_path.exists()
        other = open(sm._lock_path, "a+")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            other.close()

    # Lock released on exit — a fresh fd acquires immediately.
    other = open(sm._lock_path, "a+")
    try:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    finally:
        other.close()


def test_save_without_fcntl_falls_back(mem_dir, monkeypatch):
    """Graceful degradation: when fcntl is unavailable the save still writes
    atomically (PID-unique temp + os.replace), it just loses the
    cross-process guarantee. Exercises the no-op lock branch."""
    monkeypatch.setattr(skills_mod, "_HAS_FCNTL", False)
    sm = SkillMemory(mem_dir)
    sm.save_playbook([{"task": "x", "mistake": "y", "solution": "z"}])
    assert sm._load_playbook()[0]["task"] == "x"
    # No lock file was opened for the write (fcntl disabled) — but the save
    # is still correct.
    assert not list(mem_dir.glob("*.tmp")), "stale temp left behind"


def test_concurrent_saves_never_tear_playbook(mem_dir):
    """Two SkillMemory instances (independent RLocks, same file) hammered
    from threads: the fcntl lock serialises the write + replace so the live
    file is ALWAYS a complete, valid JSON list — never a torn temp promoted
    by os.replace — and no ``.corrupt-*`` sidecar is ever created."""
    mem = mem_dir
    sm_a = SkillMemory(mem)
    sm_b = SkillMemory(mem)

    ROUNDS = 40
    # Distinct, sizeable payloads so a torn write would be detectable.
    payload_a = [{"task": f"a{i}", "mistake": "m" * 200, "solution": "s" * 200}
                 for i in range(30)]
    payload_b = [{"task": f"b{i}", "mistake": "n" * 200, "solution": "t" * 200}
                 for i in range(30)]

    errors = []

    def worker(sm, payload):
        for _ in range(ROUNDS):
            try:
                sm.save_playbook(payload)
                loaded = sm._load_playbook()
                assert isinstance(loaded, list)
            except Exception as e:  # torn/corrupt read would surface here
                errors.append(repr(e))

    threads = [
        threading.Thread(target=worker, args=(sm_a, payload_a)),
        threading.Thread(target=worker, args=(sm_b, payload_b)),
        threading.Thread(target=worker, args=(sm_a, payload_a)),
        threading.Thread(target=worker, args=(sm_b, payload_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"torn/corrupt reads during concurrent saves: {errors[:3]}"
    # The playbook file must be valid JSON and one of the two full payloads.
    final = json.loads((mem / "skills_playbook.json").read_text())
    assert isinstance(final, list) and len(final) == 30
    assert not list(mem.glob("skills_playbook.json.corrupt-*")), (
        "a torn temp file was promoted (corrupt sidecar created)"
    )
    assert not list(mem.glob("*.tmp")), "stale temp file left behind"
