"""Regression test: file-search paths must be CONTAINER-relative.

Incident context (2026-04-26, webOS session)
--------------------------------------------
``tool_file_search`` resolved its ``filename`` arg via
``_get_safe_path`` and passed the resulting HOST-absolute path to
``sandbox_manager.execute``. ``execute`` runs the command INSIDE the
Docker container at ``workdir=/workspace``, where the host bind-mount
is the only filesystem visible. A host path like
``/Users/me/sandbox/webos/app.js`` does not exist inside the
container, so ``rg`` reported no matches and exited 0 — which the
function reported back to the agent as
``"Report: No matches found."``

The agent in the incident issued six consecutive ``rg
lockScreen.style.display webos/app.js`` calls trying to verify a
just-applied fix; all six returned empty. The wrong conclusion ("the
fix wasn't applied") drove the agent to nuke the lock screen
entirely. ~70 minutes of debugging time.

Fix
---
``tool_file_search`` now translates the host path to its
container-visible ``/workspace/...`` equivalent via
``_to_container_path`` before shlex-quoting it for the ``rg`` command.
``tool_find_files`` accepts an optional ``sandbox_dir`` and applies
the same translation when given a path that resolves under the
sandbox.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path,
    _to_container_path,
    tool_file_search,
    tool_find_files,
)


# ---------------------------------------------------------------- helpers


@pytest.fixture
def sandbox(tmp_path):
    return tmp_path


def _make_sandbox_manager(stdout: str = "", exit_code: int = 0):
    sm = AsyncMock()
    sm.execute = MagicMock(return_value=(stdout, exit_code))
    return sm


# ---------------------------------------------------------------- core helper


def test_to_container_path_basic(sandbox):
    host = sandbox / "webos" / "app.js"
    host.parent.mkdir(parents=True)
    host.touch()
    container = _to_container_path(sandbox, host)
    # Must start with /workspace and have NO trace of the host
    # sandbox-dir prefix anywhere in the resulting string.
    assert container == "/workspace/webos/app.js"
    assert str(sandbox) not in container


def test_to_container_path_root_returns_workspace(sandbox):
    # An empty / "." rel path resolves to the workspace root itself.
    container = _to_container_path(sandbox, sandbox)
    assert container == "/workspace"


def test_to_container_path_nested(sandbox):
    deep = sandbox / "a" / "b" / "c" / "file.txt"
    deep.parent.mkdir(parents=True)
    deep.touch()
    assert _to_container_path(sandbox, deep) == "/workspace/a/b/c/file.txt"


def test_to_container_path_outside_falls_back_to_workspace(sandbox, tmp_path_factory):
    # Defensive: a path NOT under sandbox_dir would normally be
    # caught by _get_safe_path upstream, but if a caller hands us
    # one anyway we return /workspace rather than raising — the
    # malformed path should not crash the search tool.
    other = tmp_path_factory.mktemp("elsewhere") / "x.py"
    other.touch()
    assert _to_container_path(sandbox, other) == "/workspace"


# ---------------------------------------------------------------- tool_file_search


@pytest.mark.asyncio
async def test_file_search_passes_container_path_not_host_path(sandbox):
    """The exact regression: rg must receive /workspace/foo.py, NOT
    the host-absolute path."""
    target = sandbox / "webos" / "app.js"
    target.parent.mkdir(parents=True)
    target.write_text("var lockScreen = document.getElementById('lock-screen');\n"
                      "lockScreen.style.display = 'none';\n")

    sm = _make_sandbox_manager(
        stdout="/workspace/webos/app.js:2:lockScreen.style.display = 'none';\n",
        exit_code=0,
    )
    result = await tool_file_search(
        "lockScreen.style.display", sandbox, "webos/app.js", sm,
    )
    assert "lockScreen.style.display" in result

    cmd = sm.execute.call_args[0][0]
    # Must contain the container path
    assert "/workspace/webos/app.js" in cmd
    # Must NOT leak the host sandbox prefix into the command — that
    # was the whole bug.
    assert str(sandbox) not in cmd, (
        f"rg command leaked host path: {cmd!r}\n"
        f"sandbox_dir={sandbox!r}"
    )


@pytest.mark.asyncio
async def test_file_search_with_no_filename_targets_active_workspace(sandbox):
    """Omitting the filename searches the ACTIVE workspace's container path
    — "/workspace" for an unscoped sandbox (same tree the old "." default
    reached), the project subdir when scoped (see the scoped test below).
    """
    sm = _make_sandbox_manager(stdout="hit.py:1:match", exit_code=0)
    await tool_file_search("anything", sandbox, None, sm)
    cmd = sm.execute.call_args[0][0]
    assert cmd.rstrip().endswith("/workspace") or "/workspace'" in cmd
    assert str(sandbox) not in cmd


@pytest.mark.asyncio
async def test_file_search_scoped_default_targets_project_dir(tmp_path):
    """Project-scoped: the no-filename default must be the PROJECT dir, not
    the sandbox root — rg ran at /workspace, so the old "." swept every
    other project's files too."""
    scoped = tmp_path / "projects" / "abc123def456"
    scoped.mkdir(parents=True)
    sm = _make_sandbox_manager(stdout="", exit_code=0)
    await tool_file_search("anything", scoped, None, sm)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/projects/abc123def456" in cmd


@pytest.mark.asyncio
async def test_file_search_handles_workspace_prefix(sandbox):
    """LLM-emitted ``/workspace/foo.py`` paths still translate
    cleanly: _get_safe_path strips the prefix, _to_container_path
    re-adds it, end result is the same /workspace/foo.py the rg
    command actually wants."""
    target = sandbox / "foo.py"
    target.touch()
    sm = _make_sandbox_manager(stdout="ok", exit_code=0)
    await tool_file_search("x", sandbox, "/workspace/foo.py", sm)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/foo.py" in cmd
    # Must not have a doubled /workspace/workspace prefix
    assert "/workspace/workspace" not in cmd


@pytest.mark.asyncio
async def test_file_search_path_translation_does_not_escape_sandbox(sandbox):
    """_get_safe_path's traversal protection still applies — a
    pathological filename can't slip through translation. The
    translation step is downstream of _get_safe_path, so any caller
    handing us "../../etc/passwd" gets rejected upstream and we
    never hit the translator."""
    # _get_safe_path resolves "../etc/passwd" to a path under sandbox
    # (because lstrip strips the leading slashes); container path
    # MUST still be /workspace-rooted.
    target = sandbox / "etc" / "passwd"
    target.parent.mkdir(parents=True)
    target.touch()
    sm = _make_sandbox_manager(stdout="", exit_code=0)
    await tool_file_search("root", sandbox, "etc/passwd", sm)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/etc/passwd" in cmd
    # Even a malformed path must not put the host root '/etc/passwd' in.
    assert " /etc/passwd" not in cmd


# ---------------------------------------------------------------- tool_find_files


@pytest.mark.asyncio
async def test_find_files_translates_path_when_sandbox_dir_provided(sandbox):
    """When sandbox_dir is passed, an absolute path inside the
    sandbox must be translated to its container-visible equivalent."""
    sub = sandbox / "src"
    sub.mkdir()
    sm = _make_sandbox_manager(
        stdout="./src/a.py\n./src/b.py\n", exit_code=0,
    )
    # LLM names the path as "src" (sandbox-relative) — the most
    # common shape; must work as before.
    await tool_find_files("*.py", sm, "src", sandbox_dir=sandbox)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/src" in cmd
    assert str(sandbox) not in cmd


@pytest.mark.asyncio
async def test_find_files_dot_path_targets_active_workspace(sandbox):
    """The default '.' path now translates to the ACTIVE workspace's
    container path — "/workspace" unscoped (the same tree "." reached, since
    find ran at the container WORKDIR), the project subdir when scoped."""
    sm = _make_sandbox_manager(stdout="./x.py\n", exit_code=0)
    await tool_find_files("*.py", sm, ".", sandbox_dir=sandbox)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace" in cmd
    assert str(sandbox) not in cmd


@pytest.mark.asyncio
async def test_find_files_scoped_dot_targets_project_dir(tmp_path):
    scoped = tmp_path / "projects" / "abc123def456"
    scoped.mkdir(parents=True)
    sm = _make_sandbox_manager(stdout="", exit_code=0)
    await tool_find_files("*.py", sm, ".", sandbox_dir=scoped)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/projects/abc123def456" in cmd


@pytest.mark.asyncio
async def test_find_files_legacy_call_without_sandbox_dir_still_works(sandbox):
    """Old callers (and the dispatcher's pre-fix shape) didn't pass
    sandbox_dir — keep the no-arg call sane: pass through whatever
    path the caller named, no translation."""
    sm = _make_sandbox_manager(stdout="./x.py\n", exit_code=0)
    await tool_find_files("*.py", sm, ".")
    cmd = sm.execute.call_args[0][0]
    # No sandbox_dir → no translation; just the literal "."
    assert "find" in cmd and "*.py" in cmd


@pytest.mark.asyncio
async def test_find_files_rejects_outside_sandbox_path(sandbox):
    sm = _make_sandbox_manager(stdout="", exit_code=0)
    res = await tool_find_files("*.py", sm, "../../../etc", sandbox_dir=sandbox)
    # Must NOT have run a find command that points at the host etc/
    assert sm.execute.call_count == 0
    assert "outside" in res.lower() or "security" in res.lower()


# ------------------------------------------- filename==pattern corruption guard


@pytest.mark.asyncio
async def test_search_heals_filename_equal_to_pattern(sandbox):
    """2026-07-17 (req AF): upstream native tool-call transport duplicated
    the search pattern into 'filename', so rg ran against
    '<workspace>/makeDraggable' — a path that cannot exist — and the turn
    was wasted. The guard drops the corrupt filename, searches the whole
    workspace, and NAMES the healing in the result."""
    sm = _make_sandbox_manager(
        stdout="/workspace/index.html:438:makeDraggable()\n", exit_code=0)
    result = await tool_file_search("makeDraggable", sandbox,
                                    "makeDraggable", sm)
    cmd = sm.execute.call_args[0][0]
    # Searched the workspace, not a file named like the pattern.
    assert cmd.rstrip().endswith("/workspace")
    assert "/workspace/makeDraggable" not in cmd
    assert "argument duplication" in result
    assert "makeDraggable()" in result


@pytest.mark.asyncio
async def test_search_heal_note_on_no_matches_too(sandbox):
    sm = _make_sandbox_manager(stdout="", exit_code=1)
    result = await tool_file_search("zZzNotThere", sandbox, "zZzNotThere", sm)
    assert "No matches found" in result
    assert "argument duplication" in result


@pytest.mark.asyncio
async def test_search_distinct_filename_untouched(sandbox):
    """A real filename must never trip the guard."""
    target = sandbox / "app.js"
    target.write_text("var makeDraggable = 1;\n")
    sm = _make_sandbox_manager(
        stdout="/workspace/app.js:1:var makeDraggable = 1;\n", exit_code=0)
    result = await tool_file_search("makeDraggable", sandbox, "app.js", sm)
    cmd = sm.execute.call_args[0][0]
    assert "/workspace/app.js" in cmd
    assert "argument duplication" not in result
