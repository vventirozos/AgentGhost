"""§4KB round 3 — pins for the defects found INSIDE round 2's fixes.

Round 2 fixed a gitfile escape and called the class closed. Round 3 proved
that was an INSTANCE fix: the model can `git init` a REAL `.git` directory and
then write inside it with the ordinary `file_system` tool — `.git/config` is a
new file, so no receipt applies — reaching five independent escapes including
**arbitrary code execution** via `core.hooksPath`.

The guard can no longer be "which files may `.git` be". It is now: hooks are
disabled on every invocation, the work tree is pinned on the command line,
every `GIT_*` override is stripped from the environment, and the repository
git ACTUALLY resolved is verified to live inside the workspace.
"""
import asyncio
import os
import pathlib
import subprocess
import time

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import (
    tool_file_system, tool_replace_text)
from ghost_agent.tools.outline import (
    TRUNCATED_KEY, build_symbol_index, render_definitions)
from ghost_agent.utils.edit_ledger import read_ledger
from ghost_agent.utils.logging import request_id_context


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))
    return h


# ---------------------------------------------------------------------------
# C1 — the escape CLASS, as a table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C2 — the batch read grounds, like the single read
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C3 — the auto-promote is a write site: gated AND stamped
# ---------------------------------------------------------------------------

COMPLETE_MODULE = '''"""A complete module."""
import os


DEFAULT_TIMEOUT = 30


class Runner:
    def __init__(self, path):
        self.path = path

    def run(self, cmd):
        return os.system(cmd)


def main():
    return Runner(".").run("true")
'''


# ---------------------------------------------------------------------------
# H1 — a FIFO must not wedge the event loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# H2 — the gate must not create a closed loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M2 / M4 — honest rendering and one unit per column
# ---------------------------------------------------------------------------

def test_a_truncated_index_warns_on_the_HIT_branch_too(tmp_path):
    """"Here are the 2 definitions" from a partly-scanned tree reads as
    complete — the more dangerous half. The caveat was computed and then used
    only on the miss branch."""
    for i in range(12):
        (tmp_path / f"f{i}.py").write_text(f"def target_{i}():\n    pass\n")
    idx = build_symbol_index(tmp_path, cap=5)
    assert TRUNCATED_KEY in idx
    hit_name = next(k for k in idx if k != TRUNCATED_KEY)
    out = render_definitions(idx, hit_name)
    assert "PARTLY scanned" in out, "a hit list from a truncated index read as complete"


def test_exactly_cap_files_is_not_truncated(tmp_path):
    """`iter_source_files` STOPS at cap, so `seen_files >= cap` over-warned
    and `> cap` could never fire. The walk now probes one file past the cap."""
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("def f():\n    pass\n")
    assert TRUNCATED_KEY not in build_symbol_index(tmp_path, cap=5)
    assert TRUNCATED_KEY in build_symbol_index(tmp_path, cap=3)


# ---------------------------------------------------------------------------
# Survivors of the combined battery — four real pin gaps
# ---------------------------------------------------------------------------


async def test_a_rolled_back_batch_logs_the_ROLLBACK_not_the_ambiguity(
        sandbox, home):
    """`telem["reason"]` is set per-block inside the loop, so a batch with one
    ambiguous block whose applied block was then ROLLED BACK logged
    `ambiguous_block` while the real verdict was the syntax rollback —
    mis-attributing rollbacks to ambiguity in the report's histogram.

    Fails in the world where telem wins over the outcome's own reason_code.
    """
    f = sandbox / "m.py"
    f.write_text("def a():\n    return 1\n\nDUP = 0\nPAD = 1\nDUP = 0\n")
    res = await tool_replace_text(
        "m.py",
        "<<<< SEARCH\n    return 1\n====\n    return (1\n>>>>\n"
        "<<<< SEARCH\nDUP = 0\n====\nDUP = 9\n>>>>",
        None, sandbox)
    assert getattr(res, "is_rejection", False), res
    (row,) = read_ledger(home=home)
    assert row["applied"] is False
    assert row["reason"] != "ambiguous_block", (
        "a syntax rollback was booked as an ambiguity")
    assert row["reason"] == res.reason_code


# ---------------------------------------------------------------------------
# Round 4 — the round-3 MECHANISMS, pinned one by one
#
# A round-3-first mutation batch killed only 5 of 15: the escape table asserts
# the OUTCOME ("nothing runs or is modified outside"), which survives removing
# any SINGLE layer because the layers are redundant. Redundancy is only a
# defence if it is deliberate and verified, so each layer is pinned here, and
# the double-mutant case is pinned too.
# ---------------------------------------------------------------------------


def test_a_truncated_index_holds_exactly_cap_files(tmp_path):
    """The walk probes `cap + 1` to DETECT truncation and must index `cap`."""
    for i in range(20):
        (tmp_path / f"f{i}.py").write_text(f"def fn{i}():\n    pass\n")
    idx = build_symbol_index(tmp_path, cap=5)
    assert TRUNCATED_KEY in idx
    names = [k for k in idx if k != TRUNCATED_KEY]
    assert len(names) == 5, f"indexed {len(names)} files, cap was 5"


def test_the_truncation_sentinel_is_never_a_lookup_result(tmp_path):
    """A model that names the sentinel must get "no definition", not a row."""
    for i in range(12):
        (tmp_path / f"f{i}.py").write_text("def f():\n    pass\n")
    idx = build_symbol_index(tmp_path, cap=5)
    out = render_definitions(idx, TRUNCATED_KEY)
    assert "No definition" in out, out
    assert "truncated" not in out.split("\n")[1] if "\n" in out else True


