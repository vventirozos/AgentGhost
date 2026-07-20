"""Regressions from the 2026-07-20 three-stack review (§4B cohort):
four execute.py fixes.

1. `_format_error` hardcoded `EXIT CODE: 1`, erasing the real code — a
   600s timeout kill (124) was indistinguishable from a program failure,
   so the model re-ran the identical >=10-min command expecting a
   different result.
2. The root-cwd retry adopted its result UNCONDITIONALLY (unlike the
   other two heal paths), so a scoped script that RAN, did side effects,
   then died on a genuine `FileNotFoundError: 'data.csv'` got its
   informative traceback REPLACED by a bogus "can't open file" from the
   root re-run.
3. `python3 <<'EOF' … urlopen(…) … EOF` EXECUTES the heredoc body, but
   the egress guard stripped it as "data" before scanning — a clean
   bypass of the loopback hard-block.
4. The "Error detected at Line N" snippet sliced the EXECUTED file's
   content at the line number of the last workspace traceback frame,
   which can belong to an IMPORTED module — wrong file's region, or an
   empty snippet when the line number exceeds the script's length.
"""

import os
import re
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.execute import (
    _EXEC_TIMEOUT_S,
    _command_probes_agent_port,
    tool_execute,
)


def _mock_mgr(returns=("ok", 0)):
    mgr = MagicMock()
    if isinstance(returns, list):
        mgr.execute = MagicMock(side_effect=returns)
    else:
        mgr.execute = MagicMock(return_value=returns)
    return mgr


# --- 1. real exit code + timeout note ----------------------------------------

async def test_timeout_kill_surfaces_124_with_note(tmp_path):
    """`timeout -k` budget kill: exit 124 must surface as 124 (not the old
    hardcoded 1) with an explicit timed-out note, so the model doesn't
    misdiagnose a bug and re-run the identical >=10-min command."""
    mgr = _mock_mgr(returns=("", 124))
    result = await tool_execute(command="python3 train_model.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 124" in result
    assert f"timed out / killed after {_EXEC_TIMEOUT_S}s" in result
    assert "EXIT CODE: 1\n" not in result
    # Downstream contract: `EXIT CODE:\s*(\d+)` still parses the real code.
    m = re.search(r"EXIT CODE:\s*(\d+)", result)
    assert m and m.group(1) == "124"


async def test_sigkill_137_gets_killed_note(tmp_path):
    mgr = _mock_mgr(returns=("", 137))
    result = await tool_execute(command="python3 big_job.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 137" in result
    assert "timed out / killed" in result


async def test_script_branch_threads_real_exit_code(tmp_path):
    """The filename/content branch reports the real code too — with the
    timeout note when the run was budget-killed."""
    mgr = _mock_mgr(returns=("", 124))
    result = await tool_execute(filename="slow.py", content="import time",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 124" in result
    assert "timed out / killed" in result


async def test_ordinary_failure_keeps_real_code_without_timeout_note(tmp_path):
    mgr = _mock_mgr(returns=("Traceback ... ValueError", 3))
    result = await tool_execute(command="python3 boom.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 3" in result
    assert "timed out" not in result


async def test_tool_level_errors_still_exit_1(tmp_path):
    """Errors minted by the tool itself (nothing ran) keep the default 1."""
    mgr = _mock_mgr()
    result = await tool_execute(filename="page.html", content="<html>",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 1" in result
    assert not mgr.execute.called


# --- 2. root-cwd retry must not clobber a real traceback ---------------------

async def test_root_retry_keeps_real_fnf_traceback_from_existing_script(tmp_path):
    """The scoped script EXISTS and ran — it died on a missing data file
    (`FileNotFoundError: 'data.csv'`). The root re-run can't even find the
    script; its bogus "can't open file" must NOT replace the informative
    traceback (mirror of the guard the other two heals use)."""
    real_tb = ("Traceback (most recent call last):\n"
               '  File "process.py", line 12, in <module>\n'
               "    with open('data.csv') as f:\n"
               "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'")
    root_bogus = ("python3: can't open file '/workspace/process.py': "
                  "[Errno 2] No such file or directory")
    mgr = _mock_mgr(returns=[(real_tb, 1), (root_bogus, 2)])
    result = await tool_execute(
        command="python3 process.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/77adb3005a92")
    assert mgr.execute.call_count == 2
    assert "data.csv" in result                      # the real error survives
    assert "can't open file '/workspace/process.py'" not in result
    assert "sandbox ROOT" not in result              # no misleading note


async def test_root_retry_success_still_adopted_with_note(tmp_path):
    fnf = ("python3: can't open file 'chart.py': "
           "[Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=[fnf, ("chart drawn", 0)])
    result = await tool_execute(
        command="python3 chart.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/77adb3005a92")
    assert mgr.execute.call_count == 2
    assert "EXIT CODE: 0" in result
    assert "chart drawn" in result
    assert "sandbox ROOT" in result


async def test_root_retry_distinct_failure_adopted_with_note(tmp_path):
    """The root run got FURTHER (a non-fnf crash): adopt it, and say where
    it ran — the lesson is about the path, not the outcome."""
    fnf = ("python3: can't open file 'parse.py': "
           "[Errno 2] No such file or directory", 2)
    crash = ("Traceback (most recent call last):\nValueError: bad literal", 1)
    mgr = _mock_mgr(returns=[fnf, crash])
    result = await tool_execute(
        command="python3 parse.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/77adb3005a92")
    assert "ValueError" in result
    assert "sandbox ROOT" in result
    assert "UNRELATED to the path" in result


# --- 3. heredoc piped into an interpreter is executed code -------------------

def test_heredoc_into_python_probe_detected():
    cmd = ("python3 <<'EOF'\n"
           "import urllib.request\n"
           "urllib.request.urlopen('http://127.0.0.1:8000/api/health')\n"
           "EOF")
    assert _command_probes_agent_port(cmd)


def test_heredoc_piped_through_cat_into_python_detected():
    cmd = ("cat <<'EOF' | python3\n"
           "import urllib.request\n"
           "urllib.request.urlopen('http://localhost:8088/v1/models')\n"
           "EOF")
    assert _command_probes_agent_port(cmd)


def test_heredoc_into_bash_probe_detected():
    cmd = ("bash <<'EOF'\n"
           "curl -s http://127.0.0.1:8000/api/health\n"
           "EOF")
    assert _command_probes_agent_port(cmd)


async def test_heredoc_into_python_probe_blocked_end_to_end(tmp_path):
    sm = MagicMock()
    res = await tool_execute(
        command=("python3 <<'EOF'\n"
                 "import urllib.request\n"
                 "urllib.request.urlopen('http://127.0.0.1:8000/')\n"
                 "EOF"),
        sandbox_dir=tmp_path, sandbox_manager=sm)
    assert "SANDBOX EGRESS BLOCKED" in res
    assert not sm.execute.called


def test_heredoc_file_write_with_agent_url_still_allowed():
    """The 2026-07-08 false positive stays fixed: a heredoc WRITTEN to a
    file (cat > …) is data, even when its content mentions the agent URL
    and a net client."""
    cmd = ("cat > app.py <<'EOF'\n"
           "import urllib.request\n"
           "GHOST_API = 'http://127.0.0.1:8000/api/chat'\n"
           "req = urllib.request.Request(GHOST_API)\n"
           "EOF")
    assert not _command_probes_agent_port(cmd)


def test_heredoc_into_innocent_python_not_flagged():
    """Interpreter heredocs WITHOUT a probe are unaffected."""
    cmd = ("python3 <<'EOF'\n"
           "print('hello')\n"
           "EOF")
    assert not _command_probes_agent_port(cmd)


# --- 4. diagnostic snippet matches the executed file's frame -----------------

async def test_snippet_uses_executed_files_own_frame(tmp_path):
    """Traceback ends in an imported module (utils.py line 40); the snippet
    must slice the EXECUTED script at ITS frame (line 2), not at line 40
    (past the end of the 3-line script → empty snippet)."""
    tb = ("Traceback (most recent call last):\n"
          '  File "main.py", line 2, in <module>\n'
          "    helper()\n"
          '  File "/workspace/utils.py", line 40, in helper\n'
          "    raise ValueError('boom')\n"
          "ValueError: boom")
    mgr = _mock_mgr(returns=(tb, 1))
    code = "from utils import helper\nhelper()\nprint('done')"
    result = await tool_execute(filename="main.py", content=code,
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "Error detected at Line 2 of 'main.py'" in result
    assert "helper()" in result          # the snippet shows the call site


async def test_foreign_frame_labeled_not_sliced(tmp_path):
    """No frame in the executed file at all: label the actual file+line
    instead of slicing the executed script at a foreign line number."""
    tb = ("Traceback (most recent call last):\n"
          '  File "/workspace/utils.py", line 40, in helper\n'
          "    raise ValueError('boom')\n"
          "ValueError: boom")
    mgr = _mock_mgr(returns=(tb, 1))
    result = await tool_execute(filename="main.py", content="print('hi')",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "Line 40 of '/workspace/utils.py'" in result
    assert "NOT in 'main.py'" in result
    # It must not pretend line 40 of the 1-line script is the error site.
    assert "Error detected at Line 40 of 'main.py'" not in result
