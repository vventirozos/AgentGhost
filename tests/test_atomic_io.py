
import pytest
import threading
import os
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.skills import SkillMemory

# Use pytest's tmp_path fixture which provides a temporary directory unique to each test invocation

def test_profile_memory_atomic_save_real_fs(tmp_path):
    # Setup
    pm = ProfileMemory(tmp_path)
    
    # Verify file created
    assert pm.file_path.exists()
    
    # We want to verify that .tmp file is used and os.replace is called.
    # We can't easily race it, but we can mock os.replace to check the call,
    # while letting the write happen to real FS.
    
    with patch("os.replace") as mock_replace:
        data = {"key": "value"}
        pm.save(data)
        
        # Verify atomic replace was called with correct paths
        expected_tmp = pm.file_path.with_suffix('.tmp')
        mock_replace.assert_called_with(expected_tmp, pm.file_path)
        
        # Verify content was written to TMP file (mock_replace prevented the move, so tmp should still exist!)
        assert expected_tmp.exists()
        assert json.loads(expected_tmp.read_text()) == data

def test_skill_memory_atomic_learn_real_fs(tmp_path):
    sm = SkillMemory(tmp_path)
    
    # skills.py now writes a PID-unique temp (skills_playbook.<pid>.tmp) so
    # concurrent processes can't clobber a shared .tmp, then os.replace's it
    # onto the live file atomically — and a finally-clause unlinks the temp on
    # any path where replace didn't consume it. So we can't leave a dangling
    # mock in place (it would delete the temp); instead capture the temp's
    # content at replace-time and let the REAL replace land the file.
    captured = {}
    real_replace = os.replace

    def _capture_and_replace(src, dst):
        captured["content"] = json.loads(Path(src).read_text())
        real_replace(src, dst)

    with patch("os.replace", side_effect=_capture_and_replace) as mock_replace:
        sm.learn_lesson("Task", "Mistake", "Solution")

        assert mock_replace.call_count == 1
        src_tmp, dst = mock_replace.call_args.args
        src_tmp = Path(src_tmp)
        assert dst == sm.file_path
        expected_tmp = sm.file_path.with_suffix(f'.{os.getpid()}.tmp')
        assert src_tmp == expected_tmp
        assert src_tmp.name != "skills_playbook.tmp"  # must be pid-qualified

    # Content written to the temp (captured before the atomic replace) and now
    # live on disk.
    assert len(captured["content"]) == 1
    assert captured["content"][0]["task"] == "Task"
    assert json.loads(sm.file_path.read_text())[0]["task"] == "Task"

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

def test_skill_memory_locking(tmp_path):
    sm = SkillMemory(tmp_path)
    mock_lock = MagicMock()
    sm._lock = mock_lock
    
    sm.learn_lesson("A", "B", "C")
    mock_lock.__enter__.assert_called()
    
    sm.get_playbook_context()
    # Should be called in fallback path (which happens when no memory_system is provided)
    mock_lock.__enter__.assert_called()
