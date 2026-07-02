"""Regressions from the 2026-07-01 chess-game trace (ghost-agent.log rounds
a4e1b33d / a4a2ce04).

Three distinct bugs made the agent loop for 10+ turns on files it had just
written:

1. `execute(filename=...)` derived the container-side run path with a
   hand-rolled strip (`lstrip("/")` + drop `sandbox/`) that had diverged from
   `_get_safe_path`'s healing. `filename="/workspace/game.py"` WROTE
   `<proj>/game.py` but RAN `python3 -u workspace/game.py` → ENOENT. The run
   path is now derived from the healed host path, so write and run always
   target the same file.

2. Shell commands referencing `/workspace/<file>` failed when a project was
   active: file_system heals `/workspace/game.py` into the project dir, but
   the container-side absolute path stayed at the sandbox ROOT. On a
   file-not-found failure the command is now retried once with `/workspace`
   remapped to the scoped project workdir.

3. `grep`-family commands exiting 1 with no output (= NO MATCHES) were
   reported as `EXIT CODE: 1`, which the agent loop scores as an execution
   strike. They now return a success-shaped result explaining the exit code.

4. The inline `python -c` → temp-file auto-conversion lost `-c`'s
   cwd-on-sys.path import semantics (`from chess_engine import Board` ran
   fine inline but ModuleNotFoundError'd from /tmp). The converted run now
   carries PYTHONPATH="$PWD...".
"""

import os
import re
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.execute import tool_execute


def _mock_mgr(returns=("ok", 0)):
    mgr = MagicMock()
    if isinstance(returns, list):
        mgr.execute = MagicMock(side_effect=returns)
    else:
        mgr.execute = MagicMock(return_value=returns)
    return mgr


# --- 1. filename run path derives from the healed host path -----------------

async def test_workspace_prefixed_filename_runs_at_healed_path(tmp_path):
    """`/workspace/game.py` with a project-scoped sandbox: file lands at
    <proj>/game.py and the container run targets `game.py` — NOT the phantom
    `workspace/game.py` (the exact 2026-07 ENOENT loop)."""
    proj = tmp_path / "projects" / "77adb3005a92"
    proj.mkdir(parents=True)
    mgr = _mock_mgr()
    result = await tool_execute(filename="/workspace/game.py",
                                content='print("hi")',
                                sandbox_dir=proj, sandbox_manager=mgr)
    assert (proj / "game.py").exists()
    ran = mgr.execute.call_args[0][0]
    assert ran == "python3 -u game.py"
    assert "workspace/game.py" not in ran
    assert "EXIT CODE: 0" in result


async def test_sandbox_prefixed_filename_runs_where_it_was_written(tmp_path):
    """`sandbox/foo.py` is honored literally by _get_safe_path (write lands at
    <sandbox>/sandbox/foo.py); the run must target the same relative path
    instead of stripping the prefix and running a different file."""
    mgr = _mock_mgr()
    await tool_execute(filename="sandbox/foo.py", content='print(1)',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert (tmp_path / "sandbox" / "foo.py").exists()
    ran = mgr.execute.call_args[0][0]
    assert ran == "python3 -u sandbox/foo.py"


async def test_plain_relative_filename_unchanged(tmp_path):
    mgr = _mock_mgr()
    await tool_execute(filename="test_chess.py", content='print(1)',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert mgr.execute.call_args[0][0] == "python3 -u test_chess.py"


async def test_traversal_still_blocked(tmp_path):
    mgr = _mock_mgr()
    result = await tool_execute(filename="../../evil.py", content='print(1)',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "Security Error" in result
    assert not mgr.execute.called


# --- 2. shell /workspace → project-workdir remap fallback -------------------

async def test_workspace_abs_path_remapped_into_project_on_fnf(tmp_path):
    """`python3 /workspace/game.py` fails file-not-found under a project scope
    → retried once with /workspace remapped to the scoped workdir."""
    fnf = ("python3: can't open file '/workspace/game.py': "
           "[Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=[fnf, ("board printed", 0)])
    result = await tool_execute(
        command="python3 /workspace/game.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/77adb3005a92")
    assert mgr.execute.call_count == 2
    retry_cmd = mgr.execute.call_args_list[1][0][0]
    assert "/workspace/projects/77adb3005a92/game.py" in retry_cmd
    assert "EXIT CODE: 0" in result
    assert "remapped" in result  # the model is told what happened


async def test_cd_workspace_prefix_remapped_on_fnf(tmp_path):
    """`cd /workspace && python3 game.py` — the bare `/workspace` cd target is
    remapped too (word-boundary match, no trailing slash required)."""
    fnf = ("python3: can't open file '/workspace/game.py': "
           "[Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=[fnf, ("ok", 0)])
    await tool_execute(
        command="cd /workspace && python3 game.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/77adb3005a92")
    retry_cmd = mgr.execute.call_args_list[1][0][0]
    assert "cd /workspace/projects/77adb3005a92 && python3 game.py" in retry_cmd


async def test_remap_skipped_when_command_already_scoped(tmp_path):
    """A command that already names the scoped dir is NOT remapped (that would
    double the prefix); the pre-existing root-cwd retry runs instead."""
    fnf = ("python3: can't open file "
           "'/workspace/projects/abc/x.py': [Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=[fnf, fnf])
    await tool_execute(
        command="python3 /workspace/projects/abc/x.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc")
    assert mgr.execute.call_count == 2
    retry_cmd = mgr.execute.call_args_list[1][0][0]
    assert "/workspace/projects/abc/projects/abc" not in retry_cmd
    # root retry re-runs the SAME command (workdir dropped, not rewritten)
    assert "python3 /workspace/projects/abc/x.py" in retry_cmd


async def test_remap_failure_keeps_original_error(tmp_path):
    """If the remapped retry ALSO can't find the file, the model sees the
    original error (the path it actually asked for), not the rewritten one."""
    fnf1 = ("python3: can't open file '/workspace/nope.py': "
            "[Errno 2] No such file or directory", 2)
    fnf2 = ("python3: can't open file "
            "'/workspace/projects/abc/nope.py': [Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=[fnf1, fnf2])
    result = await tool_execute(
        command="python3 /workspace/nope.py",
        sandbox_dir=tmp_path, sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc")
    assert "/workspace/nope.py" in result


async def test_no_remap_without_project_scope(tmp_path):
    """Unscoped (no container_workdir): behavior unchanged — single run, error
    surfaces directly (the root-retry only ever fired when scoped)."""
    fnf = ("python3: can't open file '/workspace/nope.py': "
           "[Errno 2] No such file or directory", 2)
    mgr = _mock_mgr(returns=fnf)
    result = await tool_execute(command="python3 /workspace/nope.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert mgr.execute.call_count == 1
    assert "EXIT CODE: 1" in result


# --- 3. grep exit 1 + no output = no matches, not a failure ------------------

async def test_grep_exit1_no_output_is_not_an_error(tmp_path):
    mgr = _mock_mgr(returns=("", 1))
    result = await tool_execute(command='grep -n "undo_move" ai_player.py',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 0" in result
    assert "no matches" in result.lower()


async def test_grep_exit1_docker_sentinel_output_is_not_an_error(tmp_path):
    """The docker sandbox layer substitutes a '[SYSTEM ERROR]: Process failed
    (Exit 1) with no output.' sentinel for empty output — that still counts
    as 'no output' for the no-match carve-out (the exact logged strike)."""
    mgr = _mock_mgr(
        returns=("[SYSTEM ERROR]: Process failed (Exit 1) with no output.", 1))
    result = await tool_execute(
        command='grep -n "generate_legal_moves" ai_player.py',
        sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 0" in result
    assert "no matches" in result.lower()


async def test_grep_in_trailing_pipeline_segment_carved_out(tmp_path):
    """Pipeline exit code is the LAST command's — `cd x && grep ...` exit 1
    with no output is still grep's no-match."""
    mgr = _mock_mgr(returns=("", 1))
    result = await tool_execute(command='cd src && rg "TODO" main.py',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 0" in result


async def test_non_grep_exit1_no_output_still_errors(tmp_path):
    mgr = _mock_mgr(returns=("", 1))
    result = await tool_execute(command="python3 build.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 1" in result


async def test_grep_exit2_real_error_still_errors(tmp_path):
    """grep signals genuine errors (bad regex, unreadable file) with exit 2 —
    those must still surface as failures."""
    mgr = _mock_mgr(returns=("grep: nope.py: No such file or directory", 2))
    result = await tool_execute(command='grep -n "x" nope.py',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 1" in result  # _format_error shape


async def test_grep_piped_to_non_grep_not_carved_out(tmp_path):
    """`grep ... | wc -l` — the pipeline's exit code belongs to wc; exit 1
    with no output there is NOT a grep no-match."""
    mgr = _mock_mgr(returns=("", 1))
    result = await tool_execute(command='grep -c "x" f.py | tail -n 1',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 1" in result


# --- 4. inline -c auto-convert keeps cwd import semantics --------------------

async def test_inline_py_autoconvert_carries_pythonpath(tmp_path):
    """A converted `python3 -c` body runs with PYTHONPATH="$PWD..." so local
    imports (`from chess_engine import Board`) resolve exactly as the inline
    form would have."""
    body = ("from chess_engine import Board, Piece\n"
            "board = Board()\n"
            "print('Initial board setup OK')\n"
            "moves = board.generate_all_legal_moves()\n"
            "print(f'Legal moves for white: {len(moves)}')")
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{body}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = mgr.execute.call_args[0][0]
    assert re.search(
        r'PYTHONPATH="\$PWD[^"]*" python3 /tmp/_ghost_inline_\w+\.py', ran)


async def test_inline_bash_autoconvert_has_no_pythonpath(tmp_path):
    body = "echo a; echo b; echo c; echo d"
    mgr = _mock_mgr()
    await tool_execute(command=f'bash -c "{body}"',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    ran = mgr.execute.call_args[0][0]
    assert "PYTHONPATH" not in ran
    assert re.search(r'bash /tmp/_ghost_inline_\w+\.sh', ran)
