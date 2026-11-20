"""§4UY C9/C10 review r1 — the run's status file, the runner-only rule, no trap for a command that reads statuses.

Reviewer repro (`_rev/trap_harness.py`): the r0 EXIT trap printed its failure marker on STDOUT, so `exec >
out.txt` wrote it into the user's own file; `rm` of an absent file, `lsof`, `nc -z`, `cp 2>/dev/null` and a
`timeout N` server failed a list that succeeded; PIPESTATUS read the trap's own `:`; the note named a
pipeline's LAST stage (`tee log`) instead of the program that failed.

Shell pins run the production `_bash_c` under the host's /bin/bash. On macOS that is bash 3.2; the sandbox runs
bash 5.2. Every shell pin here holds on both: the trap uses only 3.x features (`printf -v`, `+=`, `${a[@]:1}`,
`${v//[$'…']/ }`), and the PIPESTATUS comparison is against plain bash of the same version.
"""
import os
import subprocess
import tempfile

import pytest

from ghost_agent.tools import execute as ex
from ghost_agent.tools.execute import _bash_c, _normalise_exit, _read_shell_run, _take_status

BASH = "/bin/bash"


def _wrapped(command, d):
    sf = os.path.join(str(d), ex.STATUS_DIRNAME, "c" * 32)
    proc = subprocess.run(_bash_c(command, status_path=sf), shell=True, capture_output=True, text=True,
                          cwd=str(d), executable=BASH)
    return proc, sf


def _run(command, d=None):
    d = d or tempfile.mkdtemp()
    proc, sf = _wrapped(command, d)
    fails = ex.read_status_file(sf)
    code, out, _ = _read_shell_run(command, proc.returncode, proc.stdout + proc.stderr, fails)
    return code, out, fails


# --- (a) the marker never reaches the program's output ------------------------------------------------------

def test_exec_redirection_gets_no_marker_in_the_users_file(tmp_path):
    proc, _ = _wrapped("exec > out.txt; python3 -c 'raise SystemExit(3)'; echo done", tmp_path)
    assert (tmp_path / "out.txt").read_text() == "done\n"


def test_redirected_output_and_stdout_carry_no_marker(tmp_path):
    proc, sf = _wrapped("python3 -c 'raise SystemExit(3)' > log.txt; echo done", tmp_path)
    assert proc.stdout == "done\n" and "__GHOST" not in (tmp_path / "log.txt").read_text()
    assert ex.read_status_file(sf)[0][0] == 3
    assert not os.path.exists(sf)                                       # read once, removed


def test_a_job_status_file_survives_its_first_reader(tmp_path):
    (tmp_path / ex.STATUS_DIRNAME).mkdir()
    name = "d" * 32
    (tmp_path / ex.STATUS_DIRNAME / name).write_text("2:2:python3 a.py\n")
    cmd = _bash_c("python3 a.py; echo x", status_path=f"/workspace/{ex.STATUS_DIRNAME}/{name}")
    assert ex.status_failures_of_job(cmd, tmp_path) == [(2, [2], "python3 a.py")]
    assert ex.status_failures_of_job(cmd, tmp_path) == [(2, [2], "python3 a.py")]   # the wake, then collect
    assert ex.status_failures_of_job(cmd, None) == []


def test_old_status_files_age_out(tmp_path):
    d = tmp_path / ex.STATUS_DIRNAME
    d.mkdir()
    old = d / ("e" * 32)
    old.write_text("1:1:make\n")
    os.utime(old, (1, 1))
    keep = d / ("f" * 32)
    keep.write_text("1:1:make\n")
    ex.read_status_file(keep, delete=False)
    assert not old.exists() and keep.exists()


def test_a_planted_link_reads_nothing_and_unlinks_nothing_outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / ("b" * 32)
    target.write_text("9:9:python3 x.py\n")
    ws = tmp_path / "ws"
    (ws / ex.STATUS_DIRNAME).mkdir(parents=True)
    (ws / ex.STATUS_DIRNAME / ("b" * 32)).symlink_to(target)
    assert ex.read_status_file(ws / ex.STATUS_DIRNAME / ("b" * 32)) == []
    assert target.exists()


async def test_run_logs_strip_a_stored_marker(tmp_path):
    """`run logs` (runs.py) shows the job's log — a marker line an older wrapper left in it is ours."""
    from types import SimpleNamespace

    from ghost_agent.core import request_scope as RS
    from ghost_agent.tools import runs as R
    from tests.test_sandbox_job_promotion import FakeSandbox as JobSandbox

    sb = JobSandbox(tmp_path / "ws")
    (tmp_path / "labs.key").write_text("k" * 64)
    (tmp_path / "memory").mkdir()
    c = SimpleNamespace(args=SimpleNamespace(labs_host="127.0.0.1:8700", labs_key_file=str(tmp_path / "labs.key")),
                        memory_dir=tmp_path / "memory", sandbox_dir=tmp_path / "ws", sandbox_manager=sb,
                        tor_proxy="socks5://127.0.0.1:9050", current_project_id=None, conversation_key="conv1")
    tok = RS.install(RS.RequestScope(conversation_key="conv1", project_id=None, origin="owner", context=c))
    try:
        await R.tool_run(c, action="start", name="j", command="echo hi-log; printf '__GHOST_FAILED=1:make\\n'",
                         where="workspace", lifetime="job")
        import asyncio
        for _ in range(60):
            lg = await R.tool_run(c, action="logs", name="j")
            if "hi-log" in str(lg):
                break
            await asyncio.sleep(0.2)
        assert "hi-log" in str(lg) and "__GHOST_FAILED" not in str(lg)
    finally:
        RS.reset(tok)


# --- (b) only a program that runs code counts behind `;` ---------------------------------------------------

@pytest.mark.parametrize("command", [
    "rm nofile_4uy; echo ok",
    "rm -f a; rm b_4uy 2>/dev/null; echo ok",
    "cp /nonexistent_4uy . 2>/dev/null; echo ok",
    "lsof -i :59999 >/dev/null 2>&1; echo ok",
    "nc -z 127.0.0.1 1 2>/dev/null; echo ok",
    "curl -s http://127.0.0.1:1 2>/dev/null; echo ok",
    "timeout 1 python3 -c 'import time; time.sleep(5)'; echo ok",          # a bounded server / smoke run
    "ls /nonexistent_4uy; echo hi",
    "grep -q zzz /dev/null; echo after",
    "node_missing_cmd_xyz; echo ok",
])
def test_a_non_runner_failure_keeps_the_shells_status(command):
    code, out, _ = _run(command)
    assert code == 0, out


@pytest.mark.parametrize("command,expected", [
    ("python3 -c 'raise SystemExit(3)'; echo done", 3),
    ("python3 -c 'import nosuchmodule_4uy' 2>&1; echo done", 1),
    ("python3 -c 'raise SystemExit(3)' > o.log; tail -3 o.log", 3),
    ("bash -c 'exit 4'; echo z", 4),
    ("for i in 1 2; do python3 -c 'raise SystemExit(7)'; done; echo z", 7),
])
def test_a_runner_failure_behind_a_semicolon_is_the_status(command, expected):
    code, out, _ = _run(command)
    assert code == expected, out
    assert "earlier in it exited" in out


@pytest.mark.parametrize("failure,counts", [
    ((2, [2], "make all"), True), ((1, [1], "pytest -q"), True), ((1, [1], "./a.out"), True),
    ((1, [1], "gcc x.c"), True), ((1, [1], "npx tsc"), True), ((1, [1], "node app.js"), True),
    ((1, [1], "rm x"), False), ((1, [1], "pip install x"), False), ((7, [7], "curl x"), False),
    ((1, [1], "lsof -i :80"), False), ((1, [1], "./configure"), True),
])
def test_one_runner_table_decides(failure, counts):
    cmd = failure[2] + "; echo ok"
    assert bool(_normalise_exit(cmd, 0, "", [failure])) is counts


def test_a_timeout_killed_runner_is_a_smoke_run_but_its_own_crash_counts():
    assert _normalise_exit("timeout 5 python3 app.py; echo ok", 0, "", [(124, [124], "timeout 5 python3 app.py")]) == 0
    assert _normalise_exit("timeout 5 python3 app.py; echo ok", 0, "", [(1, [1], "timeout 5 python3 app.py")]) == 1


# --- (c) no trap when the command reads statuses itself -----------------------------------------------------

@pytest.mark.parametrize("command", [
    "false | true; echo ${PIPESTATUS[@]}", ". lib.sh; make", "source ./helpers.sh && pytest",
    "trap 'echo bye' EXIT; python3 a.py", "set -e; python3 a.py; echo z", "set -euo pipefail; make",
    "set -o errexit; make", "set +e; make; echo z", "exec > out.txt; make", "exec 2>&1; make", "exec &> o; make",
])
def test_a_command_reading_its_own_status_gets_no_trap(command):
    assert ex._reads_own_status(command)
    assert " ERR" not in _bash_c(command) and "__g_sf" not in _bash_c(command)


@pytest.mark.parametrize("command", ["python3 a.py; echo done", "make && ./a", "exec python3 app.py",
                                     "echo set -e", "grep -e x f; make"])
def test_an_ordinary_command_gets_the_trap(command):
    assert not ex._reads_own_status(command)
    assert " ERR" in _bash_c(command)


@pytest.mark.parametrize("command", [
    "python3 -c 'raise SystemExit(3)' | cat; echo ${PIPESTATUS[@]}",
    "false | true | (exit 5); echo \"${PIPESTATUS[0]} ${PIPESTATUS[2]}\"",
    "python3 -c 'raise SystemExit(2)' | tee /dev/null; s=(\"${PIPESTATUS[@]}\"); echo ${s[0]}",
    "python3 -c 'raise SystemExit(3)'; echo \"rc=$?\"",
])
def test_pipestatus_and_dollar_q_idioms_print_what_plain_bash_prints(command, tmp_path):
    plain = subprocess.run([BASH, "-c", command], capture_output=True, text=True, cwd=str(tmp_path))
    ours, _ = _wrapped(command, tmp_path)
    assert ours.stdout == plain.stdout


# --- (d) the note names the program that failed ---------------------------------------------------------------

def test_the_note_names_the_failing_stage_not_the_last_one():
    code, out, fails = _run("python3 -c 'raise SystemExit(2)' | tee /dev/null; echo after")
    assert code == 2
    assert "`python3 -c 'raise SystemExit(2)'` earlier in it exited 2" in out and "`tee" not in out


def test_the_note_drops_a_loop_keyword():
    code, out, _ = _run("for i in 1; do python3 -c 'raise SystemExit(7)'; done; echo z")
    assert "`python3 -c 'raise SystemExit(7)'` earlier" in out and "`do " not in out


def test_take_status_never_reads_a_printed_marker():
    out, _codes, fails = _take_status("x\n__GHOST_FAILED=3:python3 a.py\n")
    assert fails == [] and "__GHOST" not in out


# --- the job readers: the wake, the jobs landing, `run collect` ----------------------------------------------

def _job_with_status(tmp_path, line, command="python3 a.py; echo x"):
    name = "a" * 32
    (tmp_path / ex.STATUS_DIRNAME).mkdir(exist_ok=True)
    (tmp_path / ex.STATUS_DIRNAME / name).write_text(line + "\n")
    return _bash_c(command, status_path=f"/workspace/{ex.STATUS_DIRNAME}/{name}")


async def test_the_wake_reads_the_jobs_status_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import ghost_agent.main as MAIN
    import ghost_agent.sandbox.jobs as SJ
    import ghost_agent.tools.tasks as T
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda llm: False)

    class _S:
        def log_tail(self, jid, lines=60):
            return "x\n"
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: _S())
    seen = []

    async def fake_fg(ctx, body, rid, project_id=None):
        seen.append(body["messages"][0]["content"])
        return "done", None, None
    monkeypatch.setattr(MAIN, "_handle_chat_foreground", fake_fg)
    MAIN._RESUMED_JOBS.clear()
    MAIN._resume_times.clear()
    ctx = SimpleNamespace(llm_client=MagicMock(), args=SimpleNamespace(model="m"),
                          sandbox_manager=SimpleNamespace(host_workspace=tmp_path), activity_log=None,
                          agent=MagicMock())
    entry = {"id": "job-0e0e0e0e", "state": "done", "exit_code": 0, "started_by": "a1b2c3d4",
             "command": _job_with_status(tmp_path, "3:3:python3 a.py")}
    assert await MAIN._resume_after_job(ctx, entry)
    assert "EXIT CODE: 3" in seen[0]


def test_the_jobs_landing_reads_the_jobs_status_file(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ghost_agent.sandbox import jobs as J
    from ghost_agent.tools.delegate import STATUS_FAILED, _land_sandbox_row
    reg = MagicMock()
    sup = MagicMock()
    sup.sandbox = SimpleNamespace(host_workspace=tmp_path)
    sup.log_tail = lambda sid, lines=0: "x\n"
    entry = {"state": J.STATE_DONE, "exit_code": 0, "log": None,
             "command": _job_with_status(tmp_path, "2:2:python3 a.py")}
    _land_sandbox_row(reg, sup, J, MagicMock(id="r1"), "job-1", entry)
    assert reg.finish.call_args.kwargs.get("status") == STATUS_FAILED
    assert "EXIT CODE: 2" in reg.finish.call_args.kwargs.get("result", "")


async def test_run_collect_reads_the_jobs_status_file(tmp_path):
    """End to end: the job really runs (`/workspace` mapped to a tmp dir), its ERR trap writes the status file,
    `run collect` reads it."""
    import asyncio
    from types import SimpleNamespace

    from ghost_agent.core import request_scope as RS
    from ghost_agent.tools import runs as R
    from tests.test_sandbox_job_promotion import FakeSandbox as JobSandbox

    sb = JobSandbox(tmp_path / "ws")
    (tmp_path / "labs.key").write_text("k" * 64)
    (tmp_path / "memory").mkdir()
    c = SimpleNamespace(args=SimpleNamespace(labs_host="127.0.0.1:8700", labs_key_file=str(tmp_path / "labs.key")),
                        memory_dir=tmp_path / "memory", sandbox_dir=tmp_path / "ws", sandbox_manager=sb,
                        tor_proxy="socks5://127.0.0.1:9050", current_project_id=None, conversation_key="conv1")
    tok = RS.install(RS.RequestScope(conversation_key="conv1", project_id=None, origin="owner", context=c))
    try:
        await R.tool_run(c, action="start", name="j", command="python3 -c 'raise SystemExit(4)'; echo after",
                         where="workspace", lifetime="job")
        out = ""
        for _ in range(80):
            out = await R.tool_run(c, action="collect", name="j")
            if "EXIT CODE" in str(out):
                break
            await asyncio.sleep(0.2)
        assert "EXIT CODE: 4" in str(out), str(out)
    finally:
        RS.reset(tok)


# --- review r2 ------------------------------------------------------------------------------------------------

def test_a_fifo_at_the_status_path_never_blocks(tmp_path):
    """M1: a FIFO the model planted at the status path blocked the open forever — on the event loop."""
    import threading
    (tmp_path / ex.STATUS_DIRNAME).mkdir()
    name = "a" * 32
    os.mkfifo(tmp_path / ex.STATUS_DIRNAME / name)
    cmd = _bash_c("python3 a.py; echo x", status_path=f"/workspace/{ex.STATUS_DIRNAME}/{name}")
    got = []
    t = threading.Thread(target=lambda: got.append(ex.status_failures_of_job(cmd, tmp_path, delete=True)), daemon=True)
    t.start()
    t.join(3)
    assert not t.is_alive() and got == [[]]


async def test_the_foreground_read_runs_off_the_event_loop(tmp_path, monkeypatch):
    import asyncio
    import threading
    loop_thread = threading.get_ident()
    seen = []
    monkeypatch.setattr(ex, "status_failures_of_job",
                        lambda *a, **k: seen.append(threading.get_ident()) or [])
    from types import SimpleNamespace
    assert await ex._run_failures("x", SimpleNamespace(host_workspace=tmp_path)) == []
    assert seen and seen[0] != loop_thread


@pytest.mark.parametrize("command,expected", [
    ("f(){ python3 -c 'raise SystemExit(5)'; echo tail; }; f; echo after", 5),       # m5: inside a function
    ("f(){ python3 -c 'raise SystemExit(6)'; }; f; echo after", 6),
    ("python3 -m pip install nosuchpkg_4uy_zz --no-index -q 2>/dev/null; echo ok", 0),  # m2: installers
    ("python3 -m venv /nonexistent_4uy/x 2>/dev/null; echo ok", 0),
    ("make -C /nonexistent_4uy clean 2>/dev/null; echo ok", 0),
    ("make -C /nonexistent_4uy all 2>/dev/null; echo ok", 2),
])
def test_r2_trap_cases(command, expected):
    code, out, _ = _run(command)
    assert code == expected, out


def test_r2_the_note_names_the_program_inside_a_function():
    code, out, _ = _run("f(){ python3 -c 'raise SystemExit(5)'; echo tail; }; f; echo after")
    assert "`python3 -c 'raise SystemExit(5)'` earlier" in out


@pytest.mark.parametrize("cmd,runs", [
    ("python3 -m pip install x", False), ("python -m venv .venv", False), ("python3 -m ensurepip", False),
    ("python3 -m http.server 8000", False), ("/usr/bin/python3 -m pip list", False), ("make clean", False),
    ("make -C sub clean", False), ("npx kill-port 3000", False), ("npm install", False),
    ("python3 -m pytest -q", True), ("python3 -m app.main", True), ("make", True), ("make -C sub all", True),
    ("npx tsc", True), ("npm test", True), ("timeout 5 python3 app.py", True), ("./a.out", True),
])
def test_r2_runner_reading(cmd, runs):
    assert ex._runs_code(cmd) is runs


def test_r3_a_sourced_script_runs_as_plain_bash(tmp_path):
    """r3 (5): a sourced script runs in OUR shell — its own EXIT trap would replace ours and its PIPESTATUS reads
    would see the trap's. A command that sources a file gets no trap: it behaves as plain bash."""
    (tmp_path / "s.sh").write_text('false | true\necho "ps=${PIPESTATUS[*]}"\n')
    cmd = "source ./s.sh; echo end"
    assert ex._reads_own_status(cmd) and ex._reads_own_status("source lib.sh; make")
    plain = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, cwd=str(tmp_path))
    ours, _ = _wrapped(cmd, tmp_path)
    assert ours.stdout == plain.stdout == "ps=1 0\nend\n"


@pytest.mark.parametrize("cmd", ["python3 a.py; echo $?", 'python3 a.py; echo "rc=${?}"'])
def test_r3_dollar_q_keeps_the_trap_and_bash_keeps_dollar_q(cmd, tmp_path):
    """r3 (4): bash restores `$?` after the ERR trap (host bash 3.2 here; the sandbox's 5.2 is assumed the same) —
    so a command reading `$?` keeps the trap and prints what plain bash prints."""
    assert not ex._reads_own_status(cmd) and " ERR" in _bash_c(cmd)
    (tmp_path / "a.py").write_text("raise SystemExit(3)\n")
    plain = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, cwd=str(tmp_path))
    ours, _ = _wrapped(cmd, tmp_path)
    assert ours.stdout == plain.stdout and "3" in ours.stdout


def test_r3_dollarq_repro(tmp_path):
    """Reviewer's dollarq.py: the probe read with `$?` is forgiven; the LATER program's failure is the status."""
    (tmp_path / "neg.py").write_text("import sys; sys.exit(2)\n")
    (tmp_path / "app.py").write_text("print(1/0)\n")
    code, out, fails = _run('python3 neg.py; echo "rc=$?"; python3 app.py; ls >/dev/null', tmp_path)
    assert code == 1 and "rc=2" in out and "`python3 app.py` earlier in it exited 1" in out
    code, out, _ = _run('python3 neg.py; echo "rc=$?"; ls >/dev/null', tmp_path)
    assert code == 0
    # only the NEXT member's read forgives: a `$?` two members later reads the later program's status
    code, out, _ = _run('python3 neg.py; python3 -c "pass"; echo "rc=$?"', tmp_path)
    assert code == 2 and "rc=0" in out


@pytest.mark.parametrize("cmd", [
    ". .venv/bin/activate && python3 t.py; echo done",
    "source venv/bin/activate; python3 t.py; echo done",
    'source "env/bin/activate.sh" && python3 t.py; echo done',
    "source */bin/activate && python3 t.py; echo done",
    ". ~/miniconda3/etc/profile.d/conda.sh; python3 t.py; echo done",
])
def test_r3_an_environment_activator_keeps_the_status_rule(cmd, tmp_path):
    """Coordinator r3: sourcing a virtualenv `bin/activate` (or conda's conda.sh) is the common Python form and
    sets no traps — it keeps the trap, and a later failing program is still the command's status."""
    assert not ex._reads_own_status(cmd) and " ERR" in _bash_c(cmd)
    (tmp_path / "t.py").write_text("raise SystemExit(4)\n")
    for rel in (".venv/bin/activate", "venv/bin/activate", "env/bin/activate.sh", "x/bin/activate",
                "miniconda3/etc/profile.d/conda.sh"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("export VIRTUAL_ENV=1\n")
    code, out, _ = _run(cmd.replace("~/", "./"), tmp_path)
    assert code == 4 and "`python3 t.py` earlier in it exited 4" in out


@pytest.mark.parametrize("cmd", ["source lib.sh; python3 t.py; echo done", ". ./lib.sh && python3 t.py; echo done",
                                 "source venv/bin/activate_extra; python3 t.py; echo done"])
def test_r3_any_other_sourced_file_runs_as_plain_bash(cmd, tmp_path):
    assert ex._reads_own_status(cmd) and " ERR" not in _bash_c(cmd)


# ── review r4 ────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd,failed,want", [
    ("python3 t.py; RC=$?; echo done", "python3 t.py", 0),                 # captured by the next member
    ('python3 t.py; echo "rc=$?"', "python3 t.py", 0),
    ("python3 t.py\necho finished\necho $?", "python3 t.py", 1),            # read two members later: counts
    ("echo 'step: python3 t.py'; echo \"prev=$?\"; python3 t.py; echo end", "python3 t.py", 1),   # exact first
    ("python3 a.py; python3 b.py; echo $?", "python3 a.py", 1),
    ("python3 t.py && echo ok; echo $?", "python3 t.py", 1),
])
def test_r4_dollar_q_forgives_only_the_exact_member_read_next(cmd, failed, want):
    assert ex._list_failure(cmd, [(1, [1], failed)])[0] == want


def test_seam_the_status_file_is_read_blocking(tmp_path, monkeypatch):
    """§4SD: the O_NONBLOCK open switches back to blocking on a regular file before the read (virtiofs EAGAIN)."""
    d = tmp_path / ex.STATUS_DIRNAME
    d.mkdir()
    f = d / ("e" * 32)
    f.write_text("1:1,:python3 t.py\n")
    modes, real = [], os.read
    monkeypatch.setattr(os, "read", lambda fd, n: (modes.append(os.get_blocking(fd)), real(fd, n))[1])
    assert ex.read_status_file(str(f))
    assert modes == [True]


@pytest.mark.parametrize("cmd,failed,note", [
    ("for f in a; do python3 x.py | tee log; done", "tee log", "python3 x.py"),
    ("if true; then python3 x.py | tee log; fi", "tee log", "python3 x.py"),
])
def test_battery_the_note_names_the_program_not_the_keyword(cmd, failed, note):
    """C9s: a pipeline stage inside a loop/if starts with `do` / `then` — the note names the program."""
    assert ex._list_failure(cmd, [(1, [1, 0], failed)]) == (1, note)
