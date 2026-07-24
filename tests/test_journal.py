import pytest
from pathlib import Path
import json
import threading
import os
from src.ghost_agent.memory.journal import MemoryJournal

def test_journal_init(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=3)
    assert journal.file_path == tmp_path / "memory_journal.json"
    assert journal.max_capacity == 3
    assert journal.file_path.exists()
    assert json.loads(journal.file_path.read_text()) == []

def test_journal_append_spills_not_drops(tmp_path):
    """Overflowing the hot cap SPILLS the oldest to the overflow file — it is
    NOT dropped (the silent-loss bug fixed 2026-07-23)."""
    journal = MemoryJournal(tmp_path, max_capacity=2)
    journal.append("type1", {"data": 1})
    journal.append("type2", {"data": 2})

    loaded = journal.load()
    assert [i["type"] for i in loaded] == ["type1", "type2"]

    # Exceed capacity: hot keeps the newest 2; type1 spills, not lost.
    journal.append("type3", {"data": 3})
    assert [i["type"] for i in journal.load()] == ["type2", "type3"]  # hot only
    assert journal.pending_count() == 3                                # nothing lost
    # A drain returns ALL three, oldest-first.
    assert [i["type"] for i in journal.pop_all()] == ["type1", "type2", "type3"]
    assert journal.pending_count() == 0


def test_journal_burst_loses_nothing(tmp_path):
    """A back-to-back burst far past the hot cap (the B4-pilot scenario, no
    idle drain in between) must lose zero items and drain oldest-first."""
    journal = MemoryJournal(tmp_path, max_capacity=5)
    for i in range(50):
        journal.append("smart_memory", {"data": i})
    assert len(journal.load()) == 5           # hot stays bounded
    assert journal.pending_count() == 50      # every item retained
    popped = journal.pop_all()
    assert [p["data"]["data"] for p in popped] == list(range(50))  # FIFO order
    assert journal.pending_count() == 0

def test_journal_pop_all(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.append("type1", {"data": 1})
    journal.append("type2", {"data": 2})
    
    popped = journal.pop_all()
    assert len(popped) == 2
    assert popped[0]["type"] == "type1"
    
    # Should be empty now
    assert journal.load() == []
    
    # Popping empty journal
    assert journal.pop_all() == []

def test_journal_load_missing_file_returns_empty(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.file_path.unlink()
    assert journal.load() == []

def test_journal_load_reraises_transient_oserror(tmp_path):
    # A transient read failure (EACCES/EIO) must NOT read as "empty
    # journal": the next append() would atomically overwrite the intact
    # on-disk queue with a single-entry list. Fail loud instead — the
    # callers' append wrappers log a warning.
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.append("type1", {"data": 1})
    os.chmod(journal.file_path, 0o000)
    try:
        with pytest.raises(OSError):
            journal.load()
        # append() goes through load() and must propagate too, leaving
        # the on-disk queue untouched.
        with pytest.raises(OSError):
            journal.append("type2", {"data": 2})
    finally:
        os.chmod(journal.file_path, 0o600)
    loaded = journal.load()
    assert len(loaded) == 1
    assert loaded[0]["type"] == "type1"

def test_journal_load_corrupt_json_sidecars_and_recovers(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.file_path.write_text('[{"type": "smart_memory", "da')
    assert journal.load() == []
    sidecars = list(tmp_path.glob("*.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text().startswith('[{"type"')

def test_journal_load_undecodable_bytes_sidecars(tmp_path):
    # Binary garbage is corruption (not disk sickness): preserved to a
    # sidecar like bad JSON, not raised and not silently dropped.
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.file_path.write_bytes(b"\xff\xfe\x00garbage\x80")
    assert journal.load() == []
    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1
    # Journal is usable again after quarantine.
    journal.append("type1", {"data": 1})
    assert len(journal.load()) == 1

def test_journal_push_front_routes_to_overflow_head(tmp_path):
    """push_front returns requeued work to the FRONT of the logical queue (the
    overflow head): invisible to load() (hot only) but drained FIRST."""
    journal = MemoryJournal(tmp_path, max_capacity=3)
    journal.append("type3", {"data": 3})       # hot=[type3]

    journal.push_front([{"type": "type1", "data": 1}, {"type": "type2", "data": 2}])
    assert [i["type"] for i in journal.load()] == ["type3"]   # hot unchanged
    assert journal.pending_count() == 3
    # Drain order: overflow (requeued) first, then hot.
    assert [i["type"] for i in journal.pop_all()] == ["type1", "type2", "type3"]


def test_journal_push_front_never_drops_beyond_cap(tmp_path):
    """Requeuing onto a FULL hot buffer drops nothing (the old capacity-bounded
    merge discarded the surplus)."""
    journal = MemoryJournal(tmp_path, max_capacity=3)
    for i in range(3):
        journal.append(f"h{i}", {"data": i})    # hot full: h0,h1,h2
    journal.push_front([{"type": "front", "data": 99}])
    popped = journal.pop_all()
    assert [i["type"] for i in popped] == ["front", "h0", "h1", "h2"]  # front first, all kept


def test_journal_recovery_folds_inflight_to_overflow(tmp_path):
    """A crash mid-drain (in-flight staged, queue cleared) recovers every item
    to the overflow head on the next construction — at-least-once, no loss,
    even when the batch exceeds the hot cap."""
    journal = MemoryJournal(tmp_path, max_capacity=2)
    for i in range(6):
        journal.append("smart_memory", {"data": i})
    batch = journal.pop_all()                    # stages inflight, clears queue+overflow
    assert len(batch) == 6
    assert journal.inflight_path.exists()        # a real drain would ack; simulate a crash instead
    # New process: construction runs recover_inflight().
    revived = MemoryJournal(tmp_path, max_capacity=2)
    assert revived.pending_count() == 6          # all six came back
    assert [p["data"]["data"] for p in revived.pop_all()] == list(range(6))
