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
