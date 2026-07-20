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

def test_journal_append_and_cap(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=2)
    journal.append("type1", {"data": 1})
    journal.append("type2", {"data": 2})
    
    loaded = journal.load()
    assert len(loaded) == 2
    assert loaded[0]["type"] == "type1"
    assert loaded[1]["type"] == "type2"
    
    # Exceed capacity
    journal.append("type3", {"data": 3})
    loaded = journal.load()
    assert len(loaded) == 2
    assert loaded[0]["type"] == "type2"
    assert loaded[1]["type"] == "type3"

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

def test_journal_push_front(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=3)
    journal.append("type3", {"data": 3})
    
    items = [{"type": "type1", "data": 1}, {"type": "type2", "data": 2}]
    journal.push_front(items)
    
    loaded = journal.load()
    assert len(loaded) == 3
    assert loaded[0]["type"] == "type1"
    assert loaded[1]["type"] == "type2"
    assert loaded[2]["type"] == "type3"
    
    # Push front exceeding capacity
    journal.push_front([{"type": "type0", "data": 0}])
    loaded = journal.load()
    assert len(loaded) == 3
    assert loaded[0]["type"] == "type0"
    assert loaded[1]["type"] == "type1"
    assert loaded[2]["type"] == "type2"
