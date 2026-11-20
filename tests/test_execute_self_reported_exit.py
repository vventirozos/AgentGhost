"""§4HZ → §4UY C9/C10 — the run's status is the REAL exit code of the program that failed.

§4HZ (reqs 0e6cf008 + d594668e) adopted a status the OUTPUT printed (`exit=N` after a `$?` echo, bash's
`Killed` line); §4PH added output signatures for a list (a traceback, `N failed`, `npm ERR!`). §4UX-2 C9/C10:
both read prose — a failure behind `;` with no Python/pytest signature was EXIT CODE 0 (bare `echo $?`,
`gcc …; ./a`, unittest `FAILED`, a node ReferenceError, npm ≥ 10 `npm error`, `go test`, `cargo test`,
`… > out.log; tail -3 out.log`), and `0 failed, 12 passed` or a printed "exit code: 3" became failures.

Now the shell records every top-level failure itself (an ERR trap writing a status file the host reads back,
`execute.read_status_file`) and the status is the LAST failure of a program that RUNS CODE
(`_RUNNER_HEAD_RE`, review r1); a command that reads statuses itself (`$?`, PIPESTATUS, `trap`, `set -e`,
`exec >`) gets no trap. Shell claims run against a real bash — the host's (3.2 on macOS; the sandbox runs
5.2: the trap uses only 3.x features, `printf -v`, `+=`, `${a[@]:1}`) — through the production `_bash_c`.
"""
import ast
import inspect
import os
import subprocess
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools import execute as ex
from ghost_agent.tools.execute import _bash_c, _normalise_exit, _take_status, tool_execute


def _run(command, tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    sf = os.path.join(str(d), ex.STATUS_DIRNAME, "a" * 32)
    proc = subprocess.run(_bash_c(command, status_path=sf), shell=True, capture_output=True, text=True,
                          cwd=str(d), executable="/bin/bash")
    out, _codes, _ = _take_status(proc.stdout + proc.stderr)
    return _normalise_exit(command, proc.returncode, out, ex.read_status_file(sf)), out


_FAIL = "python3 -c 'import sys; sys.exit(3)'"


@pytest.mark.parametrize("command,expected", [
    # C9: a failure behind `;` / a `$?` echo, whatever the program printed
    (f"{_FAIL}; echo done", 3),
    # r1 (c): the command reads `$?` itself — no trap, it sees what it asked for
    (f"{_FAIL}; echo $?", 0),
    (f'{_FAIL}; echo "exit=$?"', 0),
    (f"{_FAIL} > /tmp/_4uy_out.log; tail -3 /tmp/_4uy_out.log", 3),
    ("python3 -c 'import nosuchmodule_xyz' 2>&1; echo done", 1),
    ("python3 -c 'import sys; print(\"FAILED (failures=1)\"); sys.exit(1)'; ls >/dev/null", 1),   # unittest
    # r1 (b): only a program that runs code counts behind `;`
    ("node_missing_cmd_xyz --run; echo ok", 0),
    ("rm /nonexistent_4uy_zz 2>/dev/null; echo ok", 0),
    ("cp /nonexistent_4uy_zz . 2>/dev/null; echo ok", 0),
    ("timeout 1 python3 -c 'import time; time.sleep(5)'; echo ok", 0),   # a bounded smoke run
    ("gcc /nonexistent_4uy_zz.c 2>/dev/null; echo ok", None),          # a compiler: counts (if installed)
    (f"for i in 1 2; do {_FAIL}; done; echo z", 3),
    # C10: printed text is never a status
    ("python3 -c 'print(\"0 failed, 12 passed\")'", 0),
    ("echo 'exit code: 3'", 0),
    ("echo 'exit=1'; echo \"$?\"", 0),
    # handled / answered statuses
    (f"{_FAIL} || true", 0),
    (f"if {_FAIL}; then :; fi; echo z", 0),
    ("ls /nonexistent_4uy_zz; echo hi", 0),
    ("grep -q zzz /dev/null; echo after", 0),
    ("python3 -c 'print(1)' | grep -q zzz; echo after", 0),
    # the d6bff7a8 shape: the error path probed on purpose, read with `$?`, then more programs
    (f'{_FAIL}; echo "exit=$?"; python3 -c "print(1)"; ls >/dev/null', 0),
    # a pipe closed early by its reader, mid-list, is forgiven like the last one
    ("yes | head -1; echo ok", 0),
    (f"{_FAIL} 2>&1 | tail -1; echo x", 3),
    # the shell's own non-zero stays
    (_FAIL, 3),
    ("false && true", 1),
])
def test_the_real_exit_code_decides(command, expected):
    code, out = _run(command)
    if expected is None:
        assert code != 0, (command, out)
    else:
        assert code == expected, (command, out)
    assert ex._FAILED_MARK not in out and ex._PIPESTATUS_MARK not in out


def test_the_wrapper_keeps_dollar_question_mark_and_pipestatus_honest():
    """The ERR trap must not change what the command itself sees: `$?` after a failure, and the stage
    statuses the grep-tail trailer reads (the trap's own commands reset PIPESTATUS)."""
    proc = subprocess.run(_bash_c(f'{_FAIL}; echo "rc=$?"'), shell=True, capture_output=True, text=True)
    assert "rc=3" in proc.stdout
    out, codes, failures = _take_status(subprocess.run(
        _bash_c(f"{_FAIL} | grep zzz"), shell=True, capture_output=True, text=True).stdout)
    assert codes == [3, 1]


def test_take_status_strips_a_legacy_marker_and_never_reads_it():
    """r1 (a): stdout is the user's — a printed marker is stripped from stored logs, never a status."""
    out, codes, failures = _take_status("a\n__GHOST_FAILED=2:make all\nlate stderr line\n")
    assert failures == []
    assert "__GHOST_FAILED" not in out and "late stderr line" in out


def _fake_exec(host_root, output, code, status_line=None):
    """A sandbox stand-in: writes the run's status file (named in the wrapped command) like the EXIT trap."""
    def run(cmd, *a, **kw):
        m = ex._STATUS_FILE_RE.search(cmd)
        if status_line and m:
            d = host_root / ex.STATUS_DIRNAME
            d.mkdir(exist_ok=True)
            (d / m.group(1).rsplit("/", 1)[-1]).write_text(status_line + "\n")
        return output, code
    return run


def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


async def test_tool_result_carries_the_earlier_programs_code_and_names_it(tmp_path):
    mgr = MagicMock()
    mgr.host_workspace = tmp_path
    mgr.execute = MagicMock(side_effect=_fake_exec(tmp_path, "Error: Cannot find module 'x'\nok\n", 0,
                                                   "1:1:node app.js"))
    result = await tool_execute(command="node app.js; echo ok", sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 1" in str(result)
    assert "`node app.js` earlier in it exited 1" in str(result)
    assert not list((tmp_path / ex.STATUS_DIRNAME).iterdir())          # a foreground run's file is read once


async def test_a_status_file_behind_a_planted_link_is_not_read(tmp_path):
    """The status directory sits in the bind mount: a link the model planted there reads nothing."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ("b" * 32)).write_text("9:9:python3 x.py\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ex.STATUS_DIRNAME).symlink_to(outside)
    assert ex.read_status_file(ws / ex.STATUS_DIRNAME / ("b" * 32)) == []
    (ws / ex.STATUS_DIRNAME).unlink()
    (ws / ex.STATUS_DIRNAME).mkdir()
    (ws / ex.STATUS_DIRNAME / ("b" * 32)).symlink_to(outside / ("b" * 32))
    assert ex.read_status_file(ws / ex.STATUS_DIRNAME / ("b" * 32)) == []
    assert (outside / ("b" * 32)).exists()


async def test_tool_result_without_a_recorded_failure_is_unchanged(tmp_path):
    result = await tool_execute(command="cat crash.log", sandbox_dir=tmp_path,
                                sandbox_manager=_mgr(("worker died: exit=1\nTraceback (most recent call last):", 0)))
    assert "--- COMMAND RESULT ---" in str(result) and "EXIT CODE: 0" in str(result)


# --- enumeration: every model-shell adoption goes through the normaliser --

def _sandbox_adoptions(tree: ast.AST):
    """(lineno, command_arg_source, exit_name) for every awaited sandbox run
    assigned in `tool_execute` — `_run_in_sandbox(mgr, CMD, …)` and
    `asyncio.to_thread(sandbox_manager.execute, CMD, …)`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Await):
            continue
        call = node.value.value
        if not isinstance(call, ast.Call):
            continue
        f = call.func
        fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        cmd_node = None
        if fname == "_run_in_sandbox" and len(call.args) >= 2:
            cmd_node = call.args[1]
        elif fname == "to_thread" and call.args:
            first = call.args[0]
            if (isinstance(first, ast.Attribute) and first.attr == "execute"
                    and getattr(first.value, "id", "") == "sandbox_manager"
                    and len(call.args) >= 2):
                cmd_node = call.args[1]
        if cmd_node is None:
            continue
        # `_run_in_sandbox(mgr, _bash_c(_remapped), …)` — the wrapper is the
        # shell framing, the argument is the model's command.
        if (isinstance(cmd_node, ast.Call) and getattr(cmd_node.func, "id", "") == "_bash_c"
                and cmd_node.args):
            cmd_node = cmd_node.args[0]
        tgt = node.targets[0]
        exit_name = None
        if isinstance(tgt, ast.Tuple) and len(tgt.elts) >= 2:
            exit_name = getattr(tgt.elts[1], "id", None)
        found.append((node.lineno, ast.unparse(cmd_node), exit_name))
    return found


def _normalisations(tree: ast.AST):
    """(lineno, exit_name) for every `_normalise_exit(_, NAME, …)` / `_read_shell_run(_, NAME, …)` call (§4UY:
    the reader of one finished run, which normalises)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("_normalise_exit", "_read_shell_run"):
            for a in node.args[1:2]:
                if isinstance(a, ast.Name):
                    out.append((node.lineno, a.id))
    return out


def _unnormalised(adoptions, normalisations):
    """Adoptions with no `_normalise_exit` on THEIR exit name between them
    and the next adoption of the same name. Per SITE, not per name: two
    heals share `_re_code`, and a normalisation after the second must not
    vouch for the first (round-22 battery: S6 survived a name-set check)."""
    bad = []
    for ln, cmd, name in adoptions:
        later_same = [l2 for l2, _, n2 in adoptions if n2 == name and l2 > ln]
        horizon = min(later_same) if later_same else float("inf")
        if not any(ln < nl < horizon and nn == name for nl, nn in normalisations):
            bad.append((ln, cmd, name))
    return bad


def _function_tree(module, name: str) -> ast.AST:
    """The FunctionDef node for `name` out of the module's parsed source."""
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_every_model_shell_adoption_is_normalised():
    """R1: the class is 'a sandbox exit code adopted in the command path'.
    Four sites today (primary, scope-flap heal, /workspace remap, root
    retry). The script path (`cmd` — a `python3 <file>` the TOOL built, no
    `$?` idiom possible, bash's own 137 already truthful) is the one
    deliberate exemption; anything else un-normalised fails here."""
    tree = _function_tree(ex, "tool_execute")
    adoptions = _sandbox_adoptions(tree)
    assert len(adoptions) >= 6, adoptions
    # A tool-BUILT command is a literal / f-string at the call (`test -f
    # {conn_file}` kernel probes); a MODEL command travels in a name
    # (`cmd_str`, `_remapped`). `cmd` is the script path's `python3 <file>`.
    exempt_cmd_names = {"cmd"}
    model_shell = [(ln, c, e) for ln, c, e in adoptions
                   if c and c.isidentifier() and c not in exempt_cmd_names]
    un = _unnormalised(model_shell, _normalisations(tree))
    assert not un, f"sandbox exits adopted without _normalise_exit: {un}"
    assert len(model_shell) == 4, model_shell
    assert {c for _, c, _ in model_shell} == {"cmd_str", "_re_cmd"}


def test_pipe_rule_is_called_only_inside_the_normaliser_and_the_list_rule():
    """Every call to the pipe rule sits inside `_normalise_exit` (the shell's code) or `_list_failure` (a
    recorded failure of an earlier pipeline) — walked on the tree, so a call in a comment counts for nothing.
    And every sandbox reader of a finished run calls `_take_status` (the marker lines are ours)."""
    module = ast.parse(inspect.getsource(ex))
    callers = {}
    for fn in ast.walk(module):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("_normalise_pipe_exit", "_list_failure"):
                callers.setdefault(node.func.id, []).append(fn.name)
    assert sorted(callers["_normalise_pipe_exit"]) == ["_list_failure", "_normalise_exit"]
    assert sorted(callers["_list_failure"]) == ["_normalise_exit", "_read_shell_run"]
