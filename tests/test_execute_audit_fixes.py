"""execute.py — 2026-07-14 audit regressions.

Covers:
  * container-absolute run path for files that resolve to the sandbox ROOT
    under a project-scoped sandbox (integration with file_system's 2026-07-14
    root-anchored _get_safe_path — the legacy lstrip fallback minted a
    phantom "workspace/x.py" relative path → ENOENT from the scoped cwd);
  * spill_large_output preserved on BOTH not-found retry paths (remap and
    root-cwd), and the root-cwd retry announcing WHERE it ran;
  * the script branch's workspace significance gate (the comment claimed
    parity with the bash branch, but every fast successful run was recorded);
  * the non-blocking loopback ground-truth note when an EXISTING script that
    matches the probe signature (agent ports + net client) is executed —
    the egress guard never saw file contents loaded from disk.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools.execute import tool_execute
from ghost_agent.workspace import WorkspaceModel

pytestmark = pytest.mark.asyncio

PID = "abc123def456"


def _scoped(tmp_path):
    root = tmp_path / "sandbox"
    sb = root / "projects" / PID
    sb.mkdir(parents=True)
    return sb, root


def _manager(responses):
    """Fake sandbox manager; `responses` is a list of (output, exit_code)
    served in order (last one repeats). Captures (cmd, kwargs) per call."""
    m = MagicMock()
    m.calls = []

    def _execute(cmd, timeout=600, **kwargs):
        m.calls.append((cmd, kwargs))
        idx = min(len(m.calls) - 1, len(responses) - 1)
        return responses[idx]

    m.execute = _execute
    return m


# ------------------------------------------------- root-anchored run path

async def test_root_anchored_file_runs_via_container_absolute_path(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "x.py").write_text("print('hi')\n")

    m = _manager([("hi", 0)])
    out = await tool_execute(
        filename="/workspace/x.py", sandbox_dir=sb, sandbox_manager=m,
        container_workdir=f"/workspace/projects/{PID}")

    assert "EXIT CODE: 0" in out
    cmd = m.calls[-1][0]
    assert "/workspace/x.py" in cmd            # container-absolute — cwd-proof
    assert " workspace/x.py" not in cmd        # NOT the phantom relative form


async def test_scoped_file_still_runs_by_bare_relative_path(tmp_path):
    sb, root = _scoped(tmp_path)
    (sb / "game.py").write_text("print('scoped')\n")

    m = _manager([("scoped", 0)])
    out = await tool_execute(
        filename="game.py", sandbox_dir=sb, sandbox_manager=m,
        container_workdir=f"/workspace/projects/{PID}")

    assert "EXIT CODE: 0" in out
    assert "game.py" in m.calls[-1][0]
    assert "/workspace/projects" not in m.calls[-1][0]  # bare relative form


# ------------------------------------------------- retry paths keep spill

async def test_workspace_remap_retry_keeps_spill(tmp_path):
    sb, root = _scoped(tmp_path)
    m = _manager([
        ("python3: can't open file '/workspace/foo.py'", 2),
        ("ran remapped", 0),
    ])
    out = await tool_execute(
        command="python3 /workspace/foo.py", sandbox_dir=sb,
        sandbox_manager=m, container_workdir=f"/workspace/projects/{PID}")

    assert len(m.calls) == 2
    assert m.calls[1][1].get("spill_large_output") is True
    assert "remapped" in out


async def test_root_cwd_retry_keeps_spill_and_announces_root(tmp_path):
    sb, root = _scoped(tmp_path)
    m = _manager([
        ("python3: can't open file 'chart.py'", 2),
        ("chart drawn", 0),
    ])
    out = await tool_execute(
        command="python3 chart.py", sandbox_dir=sb,
        sandbox_manager=m, container_workdir=f"/workspace/projects/{PID}")

    assert len(m.calls) == 2
    second_kwargs = m.calls[1][1]
    assert second_kwargs.get("spill_large_output") is True
    assert "workdir" not in second_kwargs        # root retry drops the scope
    assert "sandbox ROOT" in out                 # the model learns WHERE it ran


# ------------------------------------------------- script-branch gate

async def test_fast_successful_script_not_recorded(tmp_path):
    wm = WorkspaceModel(tmp_path)
    m = _manager([("ok", 0)])
    out = await tool_execute(
        filename="quick.py", content="print('ok')", sandbox_dir=tmp_path,
        sandbox_manager=m, workspace_model=wm)
    assert "EXIT CODE: 0" in out
    assert wm.activity.count(kind="command") == 0  # gate filtered it


async def test_failed_script_recorded(tmp_path):
    wm = WorkspaceModel(tmp_path)
    m = _manager([("Traceback ... boom", 1)])
    out = await tool_execute(
        filename="bad.py", content="print('x')", sandbox_dir=tmp_path,
        sandbox_manager=m, workspace_model=wm)
    assert "EXIT CODE: 1" in out
    cmds = wm.activity.recent(limit=5, kind="command")
    assert cmds and cmds[0].payload.get("exit_code") == 1


# ------------------------------------------------- loopback note on run

async def test_existing_probe_script_runs_with_ground_truth_note(tmp_path):
    (tmp_path / "probe.py").write_text(
        "import requests\n"
        "requests.get('http://127.0.0.1:8000/api/health')\n")
    m = _manager([("ConnectionError: refused", 1)])
    out = await tool_execute(
        filename="probe.py", sandbox_dir=tmp_path, sandbox_manager=m)

    assert len(m.calls) >= 1                     # it RAN (not blocked)
    assert "SANDBOX LOOPBACK BLIND SPOT" in out  # …but the truth rides along
    assert "mock" in out


async def test_existing_plain_script_gets_no_note(tmp_path):
    (tmp_path / "plain.py").write_text("print('hello')\n")
    m = _manager([("hello", 0)])
    out = await tool_execute(
        filename="plain.py", sandbox_dir=tmp_path, sandbox_manager=m)
    assert "EXIT CODE: 0" in out
    assert "LOOPBACK BLIND SPOT" not in out


async def test_inline_probe_content_still_hard_blocked(tmp_path):
    """The pre-existing strict rule for inline content is unchanged."""
    m = _manager([("never runs", 0)])
    out = await tool_execute(
        content="import requests; requests.get('http://127.0.0.1:8000')",
        sandbox_dir=tmp_path, sandbox_manager=m)
    assert "SANDBOX EGRESS BLOCKED" in out
    assert not m.calls                           # nothing executed
