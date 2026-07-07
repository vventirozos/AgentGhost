"""One truncation policy + spill-to-file for execute output (#10).

The effective limit on what a command could inject into context was docker's
256 KB head+tail (execute.py's own 512 KB layer was dead code behind it), so a
noisy pip-install / test-run could dump ~70 K tokens that persist in history.
Now: one shared `truncate_head_tail` helper, and the execute tool opts into
SPILL mode — small returned view + the full output written to a workspace
run-log the model can read with file_system.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.utils.text_truncate import truncate_head_tail
from ghost_agent.sandbox.docker import DockerSandbox


# ------------------------------------------------------ shared helper


def test_truncate_returns_unchanged_when_within_budget():
    text, trunc, dropped = truncate_head_tail("short", 1000)
    assert text == "short" and trunc is False and dropped == 0


def test_truncate_keeps_head_and_tail():
    body = "H" * 100 + "M" * 1000 + "T" * 100
    out, trunc, dropped = truncate_head_tail(body, 200, label="run output")
    assert trunc is True
    assert out.startswith("H")
    assert out.rstrip().endswith("T")
    assert "truncated" in out and "run output" in out
    assert dropped == len(body) - 200


def test_truncate_tail_weighted_by_default():
    body = "x" * 10000
    out, _t, _d = truncate_head_tail(body, 1000, head_frac=0.25)
    # 25% head (250) + 75% tail (750) + marker.
    head_part = out.split("[...")[0]
    assert len(head_part) <= 300


# ------------------------------------------------------ docker spill mode


def _sandbox(tmp_path):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.host_workspace = tmp_path
    sb.container = MagicMock()
    sb._last_ready_ok = 0.0
    sb._READY_TTL_S = 8.0
    return sb


def test_execute_spills_large_output_and_returns_pointer(tmp_path):
    sb = _sandbox(tmp_path)
    big = ("line of noisy build output\n" * 5000)  # ~135 KB
    sb.container.exec_run.return_value = MagicMock(
        output=big.encode(), exit_code=0)

    import unittest.mock as mock
    with mock.patch.object(sb, "ensure_running"):
        out, code = sb.execute("noisy", spill_large_output=True)

    assert code == 0
    # Returned view is small and points at the spill file.
    assert len(out) < 40 * 1024
    assert ".ghost_runs/run_" in out
    assert "saved to" in out
    # The spill file exists and holds the FULL output.
    runs = list((tmp_path / ".ghost_runs").glob("run_*.log"))
    assert len(runs) == 1
    assert runs[0].read_text().count("noisy build output") == 5000


def test_execute_small_output_not_spilled(tmp_path):
    sb = _sandbox(tmp_path)
    sb.container.exec_run.return_value = MagicMock(output=b"tiny result", exit_code=0)
    import unittest.mock as mock
    with mock.patch.object(sb, "ensure_running"):
        out, code = sb.execute("echo hi", spill_large_output=True)
    assert out == "tiny result"
    assert not (tmp_path / ".ghost_runs").exists()


def test_default_mode_uses_legacy_cap_no_spill(tmp_path):
    """Non-execute callers (rg/find/browser) keep the 256 KB head+tail and do
    NOT spill."""
    sb = _sandbox(tmp_path)
    big = ("x" * (300 * 1024))
    sb.container.exec_run.return_value = MagicMock(output=big.encode(), exit_code=0)
    import unittest.mock as mock
    with mock.patch.object(sb, "ensure_running"):
        out, code = sb.execute("rg pattern")  # no spill flag
    assert "truncated" in out
    assert ".ghost_runs" not in out
    assert not (tmp_path / ".ghost_runs").exists()


def test_execute_py_dead_512kb_layer_removed():
    """execute.py must no longer carry its own truncation layer — it delegates
    to the sandbox (spill mode)."""
    src = Path("src/ghost_agent/tools/execute.py").read_text()
    assert "512 * 1024" not in src
    assert "truncated by execute.py" not in src
    assert "spill_large_output=True" in src
