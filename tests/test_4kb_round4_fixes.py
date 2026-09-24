"""§4KB round 4 — the git escape class, closed at its precondition.

Rounds 2, 3 and 4 each closed a git escape and each was an INSTANCE fix:
a `.git` FILE, then `core.worktree`/`core.hooksPath`/`commondir`/`alternates`,
then `diff.external` and `filter.*.clean`. Git has many more keys that name a
program to run. Three rounds of denying keys one at a time is the evidence
that enumerating them does not work.

Two changes replace the enumeration:

  1. `file_system` refuses to MUTATE any path with a `.git` component. Every
     escape in every round needed that write.
  2. `git` ALLOW-LISTS the repository's own config. `execute` shares the
     workspace, so the config is still reachable there — but this agent's git
     usage is nine local subcommands on a scratch repo, and the only keys it
     needs are the ones `git init` writes. An unrecognised key, including one
     from a future git version, is refused rather than inspected.
"""
import asyncio
import os
import pathlib
import subprocess
import tempfile
import time

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import (
    tool_file_system, tool_replace_text, tool_write_file)


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. The precondition: no mutation inside `.git`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [
    ".git/config", ".git/commondir", ".git/hooks/pre-commit",
    ".git/objects/info/alternates", ".git", "sub/.git/config",
    "a/b/.git/objects/info/alternates",
])
@pytest.mark.parametrize("op", ["write", "edit", "replace", "delete"])
async def test_no_mutating_op_may_touch_dotgit(op, target, ws):
    """The class, as a table: every mutating operation x every `.git` shape.

    Fails in the world where the deny is on a filename rather than a path
    COMPONENT, or where one operation was forgotten.
    """
    kw = ({"content": "x"} if op in ("write", "replace")
          else {"old_string": "x", "new_string": "y"} if op == "edit" else {})
    res = await tool_file_system(operation=op, path=target, sandbox_dir=ws, **kw)
    assert getattr(res, "is_rejection", False), (op, target, res)
    assert res.reason_code == "dotgit_write_blocked", (op, target)


async def test_reading_dotgit_is_still_allowed(ws):
    """Diagnosis must stay possible; only MUTATION is refused."""
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    out = await tool_file_system(operation="read", path=".git/config",
                                 sandbox_dir=ws)
    assert "repositoryformatversion" in str(out)


async def test_ordinary_files_are_unaffected(ws):
    res = await tool_file_system(operation="write", path="src/app.py",
                                 content="x = 1\n", sandbox_dir=ws)
    assert "SUCCESS" in str(res)
    assert (ws / "src" / "app.py").exists()


# ---------------------------------------------------------------------------
# 2. The config allow-list — refuses keys nobody enumerated
# ---------------------------------------------------------------------------

#: Written straight to disk, as `execute` could — bypassing the `.git` block.
PROGRAM_KEYS = [
    '[diff]\n\texternal = {ev}\n',
    '[filter "p"]\n\tclean = {ev}\n',
    '[filter "p"]\n\tsmudge = {ev}\n',
    '[core]\n\tpager = {ev}\n',
    '[core]\n\tsshCommand = {ev}\n',
    '[sequence]\n\teditor = {ev}\n',
    '[credential]\n\thelper = {ev}\n',
    '[alias]\n\tx = !{ev}\n',
    '[diff "d"]\n\ttextconv = {ev}\n',
    '[uploadpack]\n\tpackObjectsHook = {ev}\n',
]


# ---------------------------------------------------------------------------
# 3. The object store: symlinked, nested, quoted
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. The FIFO wedge, at every remaining open site
# ---------------------------------------------------------------------------

MODULE = ('"""m."""\nimport os\n\n\nDEFAULT = 30\n\n\nclass R:\n'
          '    def __init__(self, p):\n        self.p = p\n\n'
          '    def run(self, c):\n        return os.system(c)\n\n\n'
          'def main():\n    return R(".").run("true")\n')


@pytest.mark.parametrize("label", ["read", "auto_promote", "symbols"])
async def test_no_open_site_wedges_on_a_fifo(label, ws):
    """Closed three times, reopened three times — `read`'s own sniff, the
    auto-promote's `write_text`, and the symbol indexer. One helper now, at
    every site that opens a model-named path."""
    f = ws / "m.py"
    os.mkfifo(f)
    try:
        t0 = time.monotonic()
        if label == "read":
            coro = tool_file_system(operation="read", path="m.py", sandbox_dir=ws)
        elif label == "auto_promote":
            coro = tool_replace_text("m.py", MODULE, None, ws)
        else:
            coro = tool_file_system(operation="symbols", name="R", sandbox_dir=ws)
        res = await asyncio.wait_for(coro, timeout=8)
        assert time.monotonic() - t0 < 5, f"{label} wedged"
        assert res is not None
    finally:
        os.unlink(f)


# ---------------------------------------------------------------------------
# 5. A lone surrogate must not destroy the file
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 7. The FLAG FILE — the override that actually reaches the live daemon
# ---------------------------------------------------------------------------


