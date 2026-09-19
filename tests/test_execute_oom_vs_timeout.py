"""§4HZ — exit 137 four seconds after launch is not a timeout.

THE LIVE FAILURE (req 0e6cf008, 2026-09-17). A plot script over a 7.4M-point
grid died with exit 137 after **4 s** (the second time; 212 s the first),
under a 600 s budget and a 4 g sandbox. `_format_error` labelled both
"(timed out / killed after 600s)" with a hint about shrinking the workload,
and the model spent four turns timing imports and `roots_legendre` before
guessing memory on its own. The elapsed time was measured at the very
line the exit code was classified, and never passed in.

World where each pin fails: the elapsed time is ignored again (both
kills read as timeouts), or the OOM note leaks the word "timeout" onto the
banner line and the strike classifier books an OOM as a transient retry.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools import execute as ex
from ghost_agent.tools.execute import _EXEC_TIMEOUT_S, _kill_is_oom, tool_execute
from ghost_agent.tools.tool_failure import FailureClass, classify_tool_failure


def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


# --- the classifier ------------------------------------------------------

@pytest.mark.parametrize("code,elapsed,expected", [
    (137, 4.0, True),                              # the live 4 s kill
    (137, 212.0, True),                            # the live first kill
    (137, _EXEC_TIMEOUT_S - 31, True),             # just inside the margin
    (137, _EXEC_TIMEOUT_S - 30, False),            # at the margin: the budget kill
    (137, _EXEC_TIMEOUT_S + 5, False),             # after -k 5s
    (137, None, False),                            # elapsed unknown → old wording
    (124, 4.0, False),                             # 124 is `timeout`'s own code
    (143, 4.0, False),                             # SIGTERM is the budget's first blow
    (1, 4.0, False),
    ("137", "4", True),                            # string-typed inputs coerce
    (137, "soon", False),                          # garbage elapsed → never OOM
])
def test_kill_is_oom_table(code, elapsed, expected):
    assert _kill_is_oom(code, elapsed) is expected


def test_margin_is_a_real_gap_under_the_budget():
    """The rule needs room: a 137 must land clearly before the budget to be
    called OOM. Zero margin would call the `-k` kill itself an OOM whenever
    the clock read a hair under 600."""
    assert 5 <= ex._OOM_KILL_MARGIN_S <= 120


# --- the tool result -----------------------------------------------------

async def test_fast_137_is_reported_as_oom_with_the_mem_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_MEM", "2g")
    result = await tool_execute(command="python3 ifs_grid_oxford.py",
                                sandbox_dir=tmp_path, sandbox_manager=_mgr(("Killed", 137)))
    assert "EXIT CODE: 137" in result
    assert "out of memory, not the time limit" in result
    assert "capped at 2g" in result
    assert "Reduce PEAK MEMORY" in result
    assert "timed out / killed after" not in result
    assert "timing imports or the computation will not explain it" in result


async def test_budget_137_keeps_the_timeout_note(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_EXEC_TIMEOUT_S", 0)   # elapsed can never be under budget − margin
    result = await tool_execute(command="python3 train.py",
                                sandbox_dir=tmp_path, sandbox_manager=_mgr(("", 137)))
    assert "timed out / killed after" in result
    assert "out of memory" not in result


async def test_124_and_143_never_take_the_oom_branch(tmp_path):
    for code in (124, 143):
        result = await tool_execute(command="python3 train.py",
                                    sandbox_dir=tmp_path, sandbox_manager=_mgr(("", code)))
        assert f"EXIT CODE: {code}" in result
        assert "timed out / killed after" in result
        assert "out of memory" not in result


async def test_oom_banner_is_not_classified_as_a_transient_retry(tmp_path):
    """The strike classifier strips the hint block but reads the banner
    line. An OOM is deterministic — re-running dies the same way — so it
    must not spend the 4-strike transient budget that "timed out" does."""
    oom = await tool_execute(command="python3 big.py",
                             sandbox_dir=tmp_path, sandbox_manager=_mgr(("Killed", 137)))
    assert classify_tool_failure(str(oom))[0] != FailureClass.RETRYABLE
    # …while the genuine budget kill stays retryable-shaped, as before.
    import ghost_agent.tools.execute as _ex
    saved = _ex._EXEC_TIMEOUT_S
    try:
        _ex._EXEC_TIMEOUT_S = 0
        to = await tool_execute(command="python3 big.py",
                                sandbox_dir=tmp_path, sandbox_manager=_mgr(("", 137)))
    finally:
        _ex._EXEC_TIMEOUT_S = saved
    assert classify_tool_failure(str(to))[0] == FailureClass.RETRYABLE


def test_mem_limit_reads_the_same_env_the_sandbox_layer_reads(monkeypatch):
    """One truth: the hint names the cap the container was created with."""
    monkeypatch.delenv("GHOST_SANDBOX_MEM", raising=False)
    assert ex._sandbox_mem_limit() == "4g"          # docker.py's default
    monkeypatch.setenv("GHOST_SANDBOX_MEM", "8g")
    assert ex._sandbox_mem_limit() == "8g"
    # The sandbox layer reads the same variable with the same default —
    # walked, not grepped: an `environ.get("GHOST_SANDBOX_MEM", <default>)`
    # call must exist in docker.py and its default must equal ours.
    import ast
    import inspect
    from ghost_agent.sandbox import docker as dk
    defaults = []
    for node in ast.walk(ast.parse(inspect.getsource(dk))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "GHOST_SANDBOX_MEM" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)):
            defaults.append(node.args[1].value)
    assert defaults == ["4g"], defaults
