"""§4IB — a real file outside /workspace is "outside", not "does not exist".

Request 21b295ef: `find` had just printed
/usr/local/lib/python3.11/site-packages/eckitlib/include/eckit/geo/grid/reduced/ReducedGaussian.h;
`file_system read` of that path answered "does not exist in the sandbox
root (/workspace …)" — a strike and a wasted turn for a file that was
there. The tool is jailed to /workspace by design; the message must say
so and hand the model the `execute` form that works.

World where each pin fails: an absolute outside path falls through to the
"does not exist" message, an inside prefix is called outside, or a read op
bypasses the one helper.
"""
import ast
import inspect

import pytest

from ghost_agent.tools import file_system as fsm
from ghost_agent.tools.file_system import (_missing_file_message, outside_workspace_message,
                                           tool_read_file)

HDR = "/usr/local/lib/python3.11/site-packages/eckitlib/include/eckit/geo/grid/reduced/ReducedGaussian.h"


@pytest.mark.parametrize("p", [HDR, "/etc/hosts", "/opt/app/main.py", "/root/x", "/home/user/a.txt"])
def test_absolute_outside_paths_get_the_outside_message(p):
    msg = outside_workspace_message(p)
    assert "OUTSIDE the sandbox workspace" in msg
    assert f"cat '{p}'" in msg and "sed -n" in msg
    assert "does not exist" not in msg


@pytest.mark.parametrize("p", ["/workspace/x.py", "/workspace", "/sandbox/a.txt", "x.py",
                               "projects/abc/x.py", "workspace/x.py", ""])
def test_inside_and_relative_paths_are_not_outside(p):
    assert outside_workspace_message(p) == ""


def test_prefix_is_a_path_segment_not_a_string_prefix():
    assert outside_workspace_message("/workspaces/evil") != ""
    assert outside_workspace_message("/sandbox-evil/x") != ""


def test_missing_file_message_routes_outside_first(tmp_path):
    msg = _missing_file_message(HDR, tmp_path)
    assert "OUTSIDE the sandbox workspace" in msg
    inside = _missing_file_message("nope.txt", tmp_path)
    assert "OUTSIDE" not in inside and "does not exist" in inside


async def test_read_of_an_outside_path_says_outside(tmp_path):
    res = await tool_read_file(HDR, tmp_path)
    assert "OUTSIDE the sandbox workspace" in str(res)
    assert "does not exist" not in str(res)


async def test_read_of_a_missing_inside_path_is_unchanged(tmp_path):
    res = await tool_read_file("missing.txt", tmp_path)
    assert "does not exist" in str(res) and "OUTSIDE" not in str(res)


def test_every_missing_file_branch_uses_the_one_helper():
    """R1: the outside check lives in `_missing_file_message`; every read-
    family op that reports a missing file goes through it (no private
    'not found' string that would skip the outside test)."""
    tree = ast.parse(inspect.getsource(fsm))
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_missing_file_message":
                callers.add(fn.name)
    assert "tool_read_file" in callers
    assert len(callers) >= 3
    outside_callers = {fn.name for fn in ast.walk(tree)
                       if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                       for c in ast.walk(fn)
                       if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "outside_workspace_message"}
    assert outside_callers == {"_missing_file_message"}
