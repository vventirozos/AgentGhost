"""A pipeline's failure is the pipeline's failure (request e0f4a8bd).

THE LIVE FAILURE, 2026-09-08. The agent ran

    python3 probe.py 2>&1 | head -200

twice. Both times probe.py raised a Traceback, and both times the tool
logged **execution ok · exit 0**, because bash returns the exit status of
the LAST command in a pipeline and `head` succeeded. So:

  * the strike counter never left 1/6 — the 6-strike budget that exists to
    stop a failing loop was never charged;
  * the failure taught the model nothing, and it re-ran the same probe;
  * the log said "✅ execution ok · Traceback (most recent call last): …",
    a status and its own contradiction on one line.

`set -o pipefail` fixes it, but naively it breaks the far commoner shape
`<producer> | head -N`, where the producer is KILLED by the closing pipe.

⚠ AND THE EXIT CODE ALONE CANNOT TELL THEM APART — the host lied about
this. On macOS a truncated Python pipeline reports 120 and standard tools
141, so a code-only exemption looked sufficient and passed every test here.
Run in the REAL sandbox (Debian, bash 5.2, CPython 3.11) the same pipeline
reports **1**: Python raises `BrokenPipeError` as an ordinary uncaught
exception, which is indistinguishable by code from a script that genuinely
failed. So the second signal is the runtime NAMING the broken pipe in
output that is captured anyway.

Every test here runs a REAL shell, and the container cases are pinned with
the shapes measured inside it. A mock cannot prove a claim about bash — and
neither, it turns out, can the wrong bash.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.tools.execute import (_bash_c, _normalise_pipe_exit,
                                       tool_execute)


def _run(command):
    """What the sandbox would see: the wrapper string, executed, then the
    tool's own normalisation over the SAME output the tool captures."""
    proc = subprocess.run(_bash_c(command), shell=True,
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    return _normalise_pipe_exit(command, proc.returncode, out), out


# --- the failure that started this ---------------------------------------

def test_a_traceback_through_head_is_a_failure():
    """THE REGRESSION, executed. World where it fails: no pipefail, and the
    exit code is `head`'s."""
    code, out = _run('python3 -c "import nosuchmodule_xyz" 2>&1 | head -200')
    assert code != 0, out
    assert "ModuleNotFoundError" in out or "Traceback" in out


def test_without_pipefail_the_same_command_reads_as_success():
    """The control that proves the test above is not vacuous: the SAME
    command, run the old way, still returns 0 today."""
    proc = subprocess.run(
        'python3 -c "import nosuchmodule_xyz" 2>&1 | head -200',
        shell=True, capture_output=True, text=True)
    assert proc.returncode == 0


@pytest.mark.parametrize("command", [
    'python3 -c "raise SystemExit(3)" | head -5',
    'python3 -c "import nosuchmodule_xyz" 2>&1 | head -5',
    'false | cat',
    'ls /definitely/not/here 2>&1 | cat',
])
def test_a_failing_stage_anywhere_in_the_pipeline_fails(command):
    code, _ = _run(command)
    assert code != 0, command


# --- …without inventing failures where there are none --------------------

@pytest.mark.parametrize("command,why", [
    ('seq 1000000 | head -3', "standard tool killed by SIGPIPE → 141"),
    ('yes | head -1', "the classic infinite producer"),
    ('python3 -c "for i in range(200000): print(i)" | head -3',
     "Python's broken-pipe flush → 120"),
    ('echo hi | head -5', "nothing truncated at all"),
    ('seq 100 | tail -2', "tail reads to the end, so nothing closes early"),
    ('seq 10 | sort | head -2', "three stages"),
])
def test_an_early_closing_downstream_is_not_a_failure(command, why):
    """The regression this fix must not cause. `… | head -N` is one of the
    commonest shapes in the log; if truncation started counting as failure,
    the 6-strike budget would drain on healthy commands — trading one
    silent-failure bug for a loud false-failure one."""
    code, _ = _run(command)
    assert code == 0, f"{command} ({why})"


#: What the REAL sandbox produces for a pipeline `head` truncated —
#: measured in the live container, 2026-09-09. The exit code is 1, the
#: same code a genuinely broken script returns.
CONTAINER_BROKEN_PIPE = "0\n1\n2\nTraceback (most recent call last):\n  File " \
    "\"<string>\", line 1, in <module>\nBrokenPipeError: [Errno 32] Broken pipe\n"
CONTAINER_REAL_FAILURE = ("Traceback (most recent call last):\n  File \"<string>\", "
                          "line 1, in <module>\nModuleNotFoundError: No module "
                          "named 'nosuchmodule_xyz'\n")


def test_the_containers_broken_pipe_exit_1_is_not_a_failure():
    """THE HOST LIED. macOS reports 120 here and the container reports 1;
    a code-only exemption passes on the laptop and turns every truncated
    `… | head` in production into a strike.

    World where it fails: the exemption reads only the exit code."""
    assert _normalise_pipe_exit("python3 x.py | head -3", 1,
                                CONTAINER_BROKEN_PIPE) == 0


def test_a_real_traceback_through_head_still_fails_in_the_container_shape():
    """The mirror, and the reason the exemption reads the error NAME rather
    than "there was a traceback": both cases print one."""
    assert _normalise_pipe_exit("python3 x.py 2>&1 | head -20", 1,
                                CONTAINER_REAL_FAILURE) == 1


def test_a_broken_pipe_outside_a_pipeline_is_still_a_failure():
    """A script that dies of a broken pipe with no pipe in the command
    talked to something else, and that is a real failure."""
    assert _normalise_pipe_exit("python3 x.py", 1, CONTAINER_BROKEN_PIPE) == 1


def test_a_bare_command_keeps_its_exit_code():
    """The exemption is narrowed to pipelines: 120 and 141 from a command
    with no pipe are real failures and stay real."""
    assert _normalise_pipe_exit("python3 x.py", 120) == 120
    assert _normalise_pipe_exit("python3 x.py", 141) == 141
    assert _normalise_pipe_exit("python3 x.py | head", 120) == 0


def test_an_OR_is_not_a_pipe():
    """`a || b` contains a "|" character and no pipeline. Reading it as one
    would hand back a free pass to every `cmd || fallback` that exits 120."""
    assert _normalise_pipe_exit("a || b", 120) == 120
    assert _normalise_pipe_exit("a || b", 141) == 141
    # …but a real pipe alongside an OR is still a pipe
    assert _normalise_pipe_exit("a || b | head -1", 141) == 0


@pytest.mark.parametrize("code", [1, 2, 3, 127, 124, 137])
def test_no_other_exit_code_is_ever_rewritten(code):
    """…when the output does not name a broken pipe."""
    assert _normalise_pipe_exit("a | head", code, "ordinary output") == code


def test_a_non_integer_exit_code_is_returned_untouched():
    """A stub manager can hand back anything; this must never raise on the
    answer path."""
    assert _normalise_pipe_exit("a | b", None) == 0
    assert _normalise_pipe_exit("a | b", "weird") == "weird"


# --- one home ------------------------------------------------------------

def test_both_shell_call_sites_go_through_the_one_wrapper():
    """The primary run and the project-path remap RETRY both build a
    `bash -c`. If only one gains pipefail, the retry keeps the blindness —
    and the retry is the path a confused model reaches most often."""
    import inspect

    from ghost_agent.tools import execute as ex
    src = inspect.getsource(ex)
    assert 'f"bash -c {shlex.quote(' not in src, \
        "a raw bash -c is built somewhere that bypasses pipefail"
    assert src.count("_bash_c(") >= 3          # definition + both call sites
    assert src.count("_normalise_pipe_exit(") >= 3


def test_the_wrapper_sets_pipefail_and_still_runs_the_command():
    wrapped = _bash_c("echo hi | head -1")
    assert wrapped.startswith("bash -c ")
    assert "set -o pipefail;" in wrapped
    assert "echo hi | head -1" in wrapped
    proc = subprocess.run(wrapped, shell=True, capture_output=True, text=True)
    assert proc.stdout.strip() == "hi"


@pytest.mark.parametrize("command", [
    "echo 'single quotes'", 'echo "double quotes"', "echo $HOME",
    "echo a && echo b", "printf 'x\\ny\\n' | grep y", "",
])
def test_quoting_survives_the_wrapper(command):
    """The command is quoted as ONE argument; a wrapper that broke quoting
    would corrupt every shell call in the agent."""
    proc = subprocess.run(_bash_c(command), shell=True,
                          capture_output=True, text=True)
    assert proc.returncode == 0, (command, proc.stderr)


# --- what the model actually reads ---------------------------------------

class LocalSandbox:
    """A sandbox manager that really runs the string it is handed."""

    container = object()
    execute_promotable_supported = False

    def execute(self, cmd, timeout=None, workdir=None, **kwargs):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (proc.stdout + proc.stderr), proc.returncode


@pytest.mark.parametrize("out,code,want_zero", [
    (CONTAINER_BROKEN_PIPE, 1, True),
    (CONTAINER_REAL_FAILURE, 1, False),
    ("", 141, True),
    ("", 120, True),
    ("clean", 0, True),
])
def test_the_container_matrix(out, code, want_zero):
    """The nine cases run inside the live container on 2026-09-09, as a
    table: exactly these pairs must map this way."""
    got = _normalise_pipe_exit("cmd | head -3", code, out)
    assert (got == 0) is want_zero, (code, out[:40], got)


@pytest.mark.asyncio
async def test_the_tool_reports_a_nonzero_exit_for_the_live_command():
    """End to end through `tool_execute`: the status line the model reads
    must contradict neither itself nor the traceback under it."""
    out = await tool_execute(
        command='python3 -c "import nosuchmodule_xyz" 2>&1 | head -200',
        sandbox_manager=LocalSandbox())
    assert "EXIT CODE: 0" not in out, out
    assert "Traceback" in out or "ModuleNotFoundError" in out


@pytest.mark.asyncio
async def test_the_tool_still_reports_success_for_a_truncated_pipeline():
    out = await tool_execute(command="seq 1000 | head -3",
                             sandbox_manager=LocalSandbox())
    assert "EXIT CODE: 0" in out, out
    assert "1" in out


class ContainerLikeSandbox:
    """Replays what the LIVE sandbox actually returned on 2026-09-09.

    The end-to-end tests above run macOS bash, where a truncated Python
    pipeline exits 120 and the code branch forgives it without ever reading
    the output — so a call site that DROPS the output argument passes them
    all. The container returns exit 1 with `BrokenPipeError` in the text,
    and only the output makes that forgivable. Measured, not imagined:

        docker exec … bash -c 'set -o pipefail; python3 -c "…" | head -3'
        → rc 1, stderr "BrokenPipeError: [Errno 32] Broken pipe"
    """

    container = object()
    execute_promotable_supported = False

    def __init__(self, output, code):
        self._output, self._code = output, code

    def execute(self, cmd, timeout=None, workdir=None, **kwargs):
        return self._output, self._code


@pytest.mark.asyncio
async def test_the_tool_forgives_the_containers_broken_pipe_shape():
    """World where it fails: the call site calls `_normalise_pipe_exit`
    without the output, so in production every truncated `… | head` costs a
    strike — invisible on a macOS test run."""
    out = await tool_execute(
        command='python3 -c "for i in range(200000): print(i)" | head -3',
        sandbox_manager=ContainerLikeSandbox(CONTAINER_BROKEN_PIPE, 1))
    assert "EXIT CODE: 0" in out, out


@pytest.mark.asyncio
async def test_the_tool_still_fails_the_containers_real_traceback_shape():
    out = await tool_execute(
        command='python3 -c "import nosuchmodule_xyz" 2>&1 | head -20',
        sandbox_manager=ContainerLikeSandbox(CONTAINER_REAL_FAILURE, 1))
    assert "EXIT CODE: 0" not in out, out
    assert "ModuleNotFoundError" in out


class TwoCallSandbox:
    """First call fails file-not-found, second (the path-remap retry)
    returns the container's broken-pipe shape."""

    container = object()
    execute_promotable_supported = False

    def __init__(self):
        self.calls = []

    def execute(self, cmd, timeout=None, workdir=None, **kwargs):
        self.calls.append(cmd)
        if len(self.calls) == 1:
            return ("python3: can't open file '/workspace/probe.py': "
                    "[Errno 2] No such file or directory"), 2
        return CONTAINER_BROKEN_PIPE, 1


@pytest.mark.asyncio
async def test_the_remap_RETRY_also_forgives_a_broken_pipe():
    """The retry is a SECOND shell call with its own exit code, and it is
    the path a confused model reaches most often. It normalised the code
    without the output, so in the container every remapped `… | head` came
    back a failure — a mutant that dropped that argument survived every
    other pin here (2026-09-09).

    Both calls must also carry pipefail: a retry without it re-creates the
    original blindness on the very path that ran because something already
    went wrong."""
    mgr = TwoCallSandbox()
    out = await tool_execute(
        command='python3 /workspace/probe.py | head -3',
        sandbox_manager=mgr,
        container_workdir="/workspace/projects/abc123def456")
    assert len(mgr.calls) == 2, mgr.calls
    assert all("set -o pipefail;" in c for c in mgr.calls), mgr.calls
    assert "/workspace/projects/abc123def456/probe.py" in mgr.calls[1]
    assert "EXIT CODE: 0" in out, out
