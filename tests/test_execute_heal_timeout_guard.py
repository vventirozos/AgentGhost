"""2026-07-22 sandbox-infra review — execute.py heal-retry safety.

The cwd-heal re-runs a command when its output looks like a wrong-cwd
file-not-found. Two shapes must NOT trigger a re-run (they'd repeat side
effects / double a 600s wall-clock):

1. A TIMEOUT KILL (124/137/143) whose accumulated output happens to mention
   ENOENT (pip/wget noise) — the command ran for its whole budget.
2. A command that RAN (has a Python traceback) and hit a missing DATA file at
   runtime — the file-not-found is post-side-effect, not a wrong-cwd miss.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.tools.execute import tool_execute, _looks_like_file_not_found


def _mock_mgr(returns):
    mgr = MagicMock()
    if isinstance(returns, list):
        mgr.execute = MagicMock(side_effect=returns)
    else:
        mgr.execute = MagicMock(return_value=returns)
    return mgr


class TestLooksLikeFileNotFound:
    def test_startup_miss_is_healable(self):
        assert _looks_like_file_not_found(
            "python3: can't open file 'x.py': [Errno 2] No such file or directory")

    def test_traceback_runtime_miss_is_not_healable(self):
        assert not _looks_like_file_not_found(
            "Traceback (most recent call last):\n"
            "  File 'p.py', line 3, in <module>\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'")


@pytest.mark.asyncio
async def test_timeout_kill_is_not_re_run(tmp_path):
    """A 124 timeout kill whose output mentions ENOENT must NOT be re-run."""
    out = ("wget: /pkg not found\n"
           "python3: can't open file 'run.py': No such file or directory")
    mgr = _mock_mgr(returns=[(out, 124), ("should-not-happen", 0)])
    result = await tool_execute(
        command="python3 run.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc")
    assert mgr.execute.call_count == 1          # NOT re-run despite the ENOENT text
    assert "should-not-happen" not in result


@pytest.mark.asyncio
async def test_ran_then_missing_data_file_is_not_re_run(tmp_path):
    """A script that RAN (traceback) and hit a missing data file must NOT be
    re-run — that would repeat its side effects."""
    tb = ("Traceback (most recent call last):\n"
          "  File 'ingest.py', line 20, in <module>\n"
          "    open('data.csv')\n"
          "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'")
    mgr = _mock_mgr(returns=[(tb, 1), ("should-not-happen", 0)])
    result = await tool_execute(
        command="python3 ingest.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc")
    assert mgr.execute.call_count == 1
    assert "data.csv" in result                 # the real error survives


@pytest.mark.asyncio
async def test_genuine_wrong_cwd_still_heals(tmp_path):
    """The legitimate case still works: a startup 'can't open file' (no
    traceback, non-timeout code) DOES trigger the root-cwd retry."""
    miss = ("python3: can't open file 'chart.py': "
            "[Errno 2] No such file or directory")
    mgr = _mock_mgr(returns=[(miss, 2), ("chart drawn", 0)])
    result = await tool_execute(
        command="python3 chart.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc")
    assert mgr.execute.call_count == 2          # re-run DID happen
    assert "chart drawn" in result
