"""The not-found message names the directory it searched (§4HD, 2026-09-15).

`_missing_file_message` said "does not exist in the current project's
sandbox … that was a DIFFERENT project/session" for EVERY directory. After
the operator cleared the ROOT sandbox (req 9b6b8757, no project active) the
model was told about a project it was not in and spent a turn on it. The
message now names the place the way the path heal, `project_download_prefix`
and `_outer_root_files_hint` already detect it. Each pin names the world it
fails in.
"""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import (
    _missing_file_message, _sandbox_where, project_download_prefix, tool_read_file)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "projects" / "0123456789ab").mkdir(parents=True)
    return tmp_path


def test_the_root_sandbox_is_named_as_the_root(root):
    """FAILS IF: the wording is the old one-size "current project's sandbox"
    — the live world: no project active, root sandbox just cleared."""
    msg = _missing_file_message("report.md", root)
    assert "does not exist in the sandbox root" in msg
    assert "the project workspace '" not in msg
    assert "current project's sandbox" not in msg


def test_a_project_workspace_is_named_with_its_prefix(root):
    """FAILS IF: the scoped branch is lost or names the wrong id."""
    msg = _missing_file_message("report.md", root / "projects" / "0123456789ab")
    assert "does not exist in the project workspace 'projects/0123456789ab/'" in msg
    assert "sandbox root" not in msg


def test_the_empty_line_names_the_place_once(root):
    """FAILS IF: the EMPTY sentence re-states the directory (the first
    draft read "The sandbox root (…) is currently EMPTY" after a sentence
    that had just said so) — or loses the word the loop-breaker pins."""
    msg = _missing_file_message("report.md", root)
    assert msg.count("sandbox root") == 1
    assert "It is currently EMPTY" in msg


def test_the_stale_hint_explanation_covers_a_cleared_sandbox(root):
    """FAILS IF: the message still blames only a DIFFERENT project — the
    live file was in THIS directory an hour earlier and was removed."""
    msg = _missing_file_message("report.md", root)
    assert "DIFFERENT project" in msg
    assert "since been removed" in msg


def test_where_agrees_with_the_download_prefix_detection(root):
    """FAILS IF: `_sandbox_where` grows its own detection — the three
    existing detectors (path heal, download prefix, outer-root hint) all
    key on `parent.name == "projects"` and this one must too."""
    for d in (root, root / "projects" / "0123456789ab", root / "projects"):
        assert (project_download_prefix(d) != "") == ("the project workspace '" in _sandbox_where(d))


@pytest.mark.asyncio
async def test_the_read_tool_delivers_the_root_wording(root):
    """FAILS IF: the call site bypasses the helper — the model reads the
    tool result, not the function."""
    res = await tool_read_file("report.md", root)
    assert "does not exist in the sandbox root" in res


def test_no_caller_in_file_system_spells_the_old_wording():
    """FAILS IF: any string literal in file_system.py reintroduces the
    one-size phrase (AST enumeration, not a text grep — comments may cite it)."""
    tree = ast.parse(Path(FS.__file__).read_text())
    docstrings = {id(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings]
    assert not [s for s in literals if "current project's sandbox" in s]
