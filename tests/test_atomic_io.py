
import pytest
import threading
import os
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from ghost_agent.memory.profile import ProfileMemory

# Use pytest's tmp_path fixture which provides a temporary directory unique to each test invocation

def test_profile_memory_atomic_save_real_fs(tmp_path):
    # Setup
    pm = ProfileMemory(tmp_path)
    
    # Verify file created
    assert pm.file_path.exists()
    
    # We want to verify that .tmp file is used and os.replace is called.
    # We can't easily race it, but we can mock os.replace to check the call,
    # while letting the write happen to real FS.
    
    # §4MD: the shared writer (utils/json_store.write_json_atomic) — a UNIQUE
    # temp file in the same directory, fsync'd, then os.replace onto the store
    seen = {}
    real_replace = os.replace

    def spy(tmp_file, dst):
        seen["src"], seen["dst"] = Path(tmp_file), Path(dst)
        with open(tmp_file, encoding="utf-8") as _fh:
            seen["text"] = _fh.read()
        return real_replace(tmp_file, dst)
    with patch("os.replace", side_effect=spy):
        data = {"key": "value"}
        pm.save(data)
    assert seen["dst"] == pm.file_path and seen["src"].parent == pm.file_path.parent
    assert seen["src"].name.endswith(".tmp") and seen["src"] != pm.file_path.with_suffix(".tmp")
    assert json.loads(seen["text"])["key"] == "value"
    assert not list(pm.file_path.parent.glob("*.tmp"))       # nothing left behind


def test_profile_memory_locking(tmp_path):
    # Verify that the lock is acquired
    
    # We need to access the lock instance on the object
    pm = ProfileMemory(tmp_path)
    
    # Mock the lock context manager *on the instance*
    # Since we can't easily verify "it was locked" after the fact without a spy,
    # we'll replace the lock with a MagicMock wrapping the real lock? 
    # Or just a MagicMock.
    
    mock_lock = MagicMock()
    pm._lock = mock_lock
    
    pm.save({"a": 1})
    mock_lock.__enter__.assert_called()
    
    pm.load()
    mock_lock.__enter__.assert_called()

