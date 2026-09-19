"""§4HZ — the status the run REPORTED beats the 0 the shell returned.

THE LIVE FAILURE (reqs 0e6cf008 + d594668e, 2026-09-17). The model ends
commands with `; echo "exit=$?"` / `echo "=== exit: $? ==="`. The LAST
command is then an `echo`, bash returns 0, and the tool logged
**execution ok · exit 0** over

  * a Python traceback followed by `exit=1`;
  * `bash: line 1:  1008 Killed  timeout 400 python3 …` (an OOM mid-run);
  * `ERROR: No matching distribution found … === exit: 1 ===`.

Consequences measured in the log: the strike counter fell from 4/6 to 1/6
on one such line, System 3's chosen recovery ("capture the full pip
output") was booked a success while changing nothing, and 8 of the day's
30 "execution ok" lines carried failure text in their body.

`set -o pipefail` (§4GE) cannot see this — there is no pipe. The echo
itself is the exact signal: the command text contains `$?` and the output
contains the number it printed. `cat crash.log` printing "exit=1" has no
`$?` and stays what it is: data.

Shell claims are executed against a real bash, as in
tests/test_execute_pipefail.py; the container's literal strings from the
log are pinned as unit rows beside them.
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
from ghost_agent.tools.execute import (_bash_c, _normalise_exit,
                                       _self_reported_exit, tool_execute)


def _run(command):
    proc = subprocess.run(_bash_c(command), shell=True, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    return _normalise_exit(command, proc.returncode, out), out


# --- the live strings ----------------------------------------------------

@pytest.mark.parametrize("command,code,output,expected", [
    ('pip install ecmwf-api 2>&1\necho "=== exit: $? ==="', 0,
     "ERROR: No matching distribution found for ecmwf-api\n=== exit: 1 ===", 1),
    ('cd /workspace && timeout 400 python3 -u ifs_grid_oxford.py 2>&1; echo "exit=$?"', 0,
     "T205: Nlon=308 …\nbash: line 1:  1008 Killed                  timeout 400 python3 -u ifs_grid_oxford.py 2>&1\nexit=137", 137),
    ('cd /workspace && timeout 300 python3 - <<\'PYEOF\'\n...\nPYEOF\necho "exit=$?"', 0,
     "near count: 149\nTraceback (most recent call last):\n  File \"<stdin>\", line 37\nIndexError: too many indices\nexit=1", 1),
    # bash's own SIGKILL report, no echo at all (the 4-second OOM)
    ("cd /workspace && python3 ifs_grid_oxford.py", 0,
     "bash: line 1:   807 Killed                  python3 ifs_grid_oxford.py", 137),
    # the last echo wins: a multi-probe script that summarises with 0
    ('false; echo exit=$?; true; echo exit=$?', 0, "exit=1\nexit=0", 0),
    # honest zero
    ('python3 x.py; echo "exit=$?"', 0, "done\nexit=0", 0),
    # no `$?` in the command → "exit=1" is data
    ("cat crash.log", 0, "… the job said exit=1 and stopped", 0),
    ("grep -n 'exit code: 2' build.log", 0, "12: exit code: 2", 0),
    # a real non-zero is never lowered
    ('python3 x.py; echo "exit=$?"', 2, "exit=0", 2),
    # out-of-range or malformed reports are ignored
    ('x; echo exit=$?', 0, "exit=999", 0),
    ('x; echo exit=$?', 0, "exit=", 0),
    # None / garbage exit codes pass through untouched
    ('x; echo exit=$?', None, "exit=1", 1),
    ('x; echo exit=$?', "abc", "exit=1", "abc"),
])
def test_self_reported_exit_table(command, code, output, expected):
    assert _self_reported_exit(command, code, output) == expected


# --- against a real shell ------------------------------------------------

def test_traceback_then_echo_is_a_failure():
    code, out = _run('python3 -c "import nosuchmodule_xyz" 2>&1; echo "exit=$?"')
    assert "ModuleNotFoundError" in out and "exit=1" in out
    assert code == 1


def test_banner_variant_is_a_failure():
    code, out = _run('false 2>&1\necho "=== exit: $? ==="')
    assert code == 1


def test_honest_success_stays_zero():
    code, _ = _run('true; echo "exit=$?"')
    assert code == 0


def test_data_that_looks_like_a_status_is_left_alone(tmp_path):
    p = tmp_path / "crash.log"
    p.write_text("worker died: exit=1\n")
    code, out = _run(f"cat {p}")
    assert "exit=1" in out and code == 0


def test_sigkilled_child_is_reported_as_137_even_when_the_shell_says_0():
    """bash prints its job line for a SIGKILLed foreground child and then
    runs the trailing echo, which returns 0."""
    code, out = _run('python3 -c "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"; echo tail')
    assert "Killed" in out, out
    assert code == 137


def test_pipe_forgiveness_applies_to_the_echoed_status_too():
    """Under pipefail the echoed `$?` is the PIPELINE's raw status, so an
    early-closed `| head -1; echo "exit=$?"` prints the SIGPIPE'd
    producer's code (141 / 120 / 1 + BrokenPipeError). The self-report gets
    the pipe rule's forgiveness by the same rule the shell code got."""
    code, out = _run('python3 -c "print(\'x\\n\' * 200000)" | head -1; echo "exit=$?"')
    assert "exit=0" not in out          # the echo really did report the SIGPIPE
    assert code == 0


# --- the tool result -----------------------------------------------------

def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


async def test_tool_result_carries_the_reported_code(tmp_path):
    result = await tool_execute(
        command='pip install ecmwf-api 2>&1\necho "=== exit: $? ==="',
        sandbox_dir=tmp_path,
        sandbox_manager=_mgr(("ERROR: No matching distribution found\n=== exit: 1 ===", 0)))
    assert "--- EXECUTION RESULT ---" in result
    assert "EXIT CODE: 1" in result
    assert "COMMAND RESULT" not in result


async def test_tool_result_without_the_echo_is_unchanged(tmp_path):
    result = await tool_execute(
        command="cat crash.log", sandbox_dir=tmp_path,
        sandbox_manager=_mgr(("worker died: exit=1", 0)))
    assert "--- COMMAND RESULT ---" in result and "EXIT CODE: 0" in result


async def test_tool_result_for_a_shell_reported_kill_is_137_with_the_oom_note(tmp_path):
    result = await tool_execute(
        command="python3 ifs_grid_oxford.py", sandbox_dir=tmp_path,
        sandbox_manager=_mgr(("bash: line 1:   807 Killed                  python3 ifs_grid_oxford.py", 0)))
    assert "EXIT CODE: 137" in result
    assert "out of memory" in result


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
    """(lineno, exit_name) for every `_normalise_exit(_, NAME, _)` call."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_normalise_exit":
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
    assert {c for _, c, _ in model_shell} == {"cmd_str", "_remapped"}


def test_pipe_rule_is_called_only_inside_the_one_normaliser():
    """Every call to the pipe rule in the module sits inside
    `_normalise_exit` (shell code, then the echoed code), which also calls
    the self-report — walked on the tree, so a call hidden in a comment or
    a string counts for nothing."""
    module = ast.parse(inspect.getsource(ex))
    callers = {}
    for fn in ast.walk(module):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in (
                    "_normalise_pipe_exit", "_self_reported_exit"):
                callers.setdefault(node.func.id, []).append(fn.name)
    assert callers["_normalise_pipe_exit"] == ["_normalise_exit", "_normalise_exit"]
    assert callers["_self_reported_exit"] == ["_normalise_exit"]
