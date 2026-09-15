"""§4GI (2026-09-13): ONE implementation of the rules for numbers the sandbox
can write — pid floor, row validation, three-valued probe, dash-safe kill —
shared by the two registries (`sandbox/jobs.py`, `sandbox/services.py`),
plus the job sentinel nonce that survives a restart.

The two files were twins hardened in different rounds: jobs got the pid
floor, row validation, `_probe_inconclusive` and the dash-safe kill in
§4DX; services kept `kill -TERM -- -<pid> || kill -TERM <pid>` (the `--`
form is a dash syntax error, so the fallback signalled the plain pid — for a
planted pid-1 row, docker-init) and read any infra fault as "dead" (rows
popped, ports re-issued). Pre-fix worlds are named per pin; the AST
enumeration fails if either file grows its own copy of any rule again.
"""
import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ghost_agent.sandbox import jobs as jobs_mod
from ghost_agent.sandbox import registry_guard as rg
from ghost_agent.sandbox import services as services_mod
from ghost_agent.sandbox.jobs import SandboxJobSupervisor
from ghost_agent.sandbox.services import ServiceSupervisor
from tests.test_sandbox_services import FakeSandbox, happy_handler


# ── the shared rules ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("pid,expect", [
    (None, None), (0, None), (1, None), ("1", None), (True, None), (-5, None),
    ("abc", None), (rg.PID_MAX, None), (rg.PID_MAX + 7, None),
    (2, 2), ("4242", 4242), (rg.PID_MAX - 1, rg.PID_MAX - 1),
])
def test_valid_pid(pid, expect):
    assert rg.valid_pid(pid) == expect


@pytest.mark.parametrize("port,expect", [
    (None, None), (0, None), (65536, None), ("x", None), (True, None), (8100, 8100), ("8101", 8101),
])
def test_valid_port(port, expect):
    assert rg.valid_port(port) == expect


@pytest.mark.parametrize("row,require_pid,ok", [
    ({"pid": 4242, "port": 8100, "name": "web"}, True, True),
    ({"pid": None, "port": 8100, "name": "web"}, False, True),
    ({"pid": None, "port": 8100, "name": "web"}, True, False),
    ({"pid": 1, "port": 8100, "name": "web"}, False, False),
    ({"pid": 0, "name": "web"}, False, False),
    ({"pid": 4242, "port": 70000, "name": "web"}, False, False),
    ({"pid": 4242, "name": "../web"}, False, False),
    ({"pid": 4242, "name": "we--b"}, False, False),
    ({"pid": 4242, "container_id": "x" * 200}, False, False),
    ("not a dict", False, False),
])
def test_validate_row(row, require_pid, ok):
    assert (rg.validate_row(row, require_pid=require_pid) is None) is ok


def test_probe_inconclusive_names_the_two_measured_shapes():
    assert rg.probe_inconclusive("[SANDBOX INFRA ERROR] daemon wedged", 1)
    assert rg.probe_inconclusive("[SYSTEM ERROR]: Process failed", 124)
    assert rg.probe_inconclusive("", 137) and rg.probe_inconclusive("", 143)
    assert not rg.probe_inconclusive("", 1)
    assert not rg.probe_inconclusive("", 0)


def test_pid_state_three_values_and_the_floor():
    def alive(cmd, timeout=15): return ("", 0)
    def dead(cmd, timeout=15): return ("", 1)
    def infra(cmd, timeout=15): return ("[SANDBOX INFRA ERROR] x", 1)
    assert rg.pid_state(alive, 4242) is True
    assert rg.pid_state(dead, 4242) is False
    assert rg.pid_state(infra, 4242) is None
    calls = []
    assert rg.pid_state(lambda c, timeout=15: calls.append(c) or ("", 0), 1) is False
    assert calls == []                          # pid 1 is never even probed


def test_kill_tree_script_is_dash_safe_and_names_both_signals():
    script = rg.kill_tree_script(4242)
    assert "-- -" not in script
    assert "S=4242;" in script
    assert 'kill -"$1" -$S' in script and "sig TERM;" in script and "sig KILL;" in script
    assert 'kill -"$1" $S' in script            # plain-pid fallback
    with pytest.raises(ValueError):
        rg.kill_tree_script(1)


def test_kill_tree_refuses_the_floor_with_a_log_line_and_sends_nothing():
    sent, logged = [], []
    for bad in (0, 1, "1", None, "x", rg.PID_MAX):
        assert rg.kill_tree(lambda c, timeout=30: sent.append(c) or ("", 0), bad,
                            log=logged.append) is False
    assert sent == [] and len(logged) == 6

    # §4GK round 4: kill_tree now answers "the tree is GONE", not "a signal
    # was sent", so the fake has to answer the liveness probe too. `kill -0`
    # exiting NON-zero is how the container says "no such process".
    def _exec_dead(cmd, timeout=30):
        sent.append(cmd)
        return ("", 0) if "S=4242;" in cmd else ("", 1)
    assert rg.kill_tree(_exec_dead, 4242) is True
    assert "S=4242;" in sent[0]


def test_kill_tree_reports_FALSE_when_the_process_survives_the_kill():
    """§4GK round 4. The old body discarded the exec result and returned an
    unconditional True once the pid passed the floor, so a process that
    survived TERM+KILL — or an exec that failed outright — read as a clean
    stop. `services.py` believed it, unlinked the pidfile and dropped the
    registry row while the process kept running and holding its port.

    Fails in any tree where kill_tree answers "a signal was sent"."""
    # §4GK round 5: the verdict comes from the kill script itself, printed by
    # the SAME shell that did the killing — one exec, and no window in which
    # the process can die (or its pid be reused) between a kill and a
    # separate probe.
    assert rg.kill_tree(lambda c, timeout=30: (rg.SURVIVED_MARKER, 0), 4242) is False
    assert rg.kill_tree(lambda c, timeout=30: (rg.KILLED_MARKER, 0), 4242) is True
    # an exec that fails outright never claims success
    assert rg.kill_tree(lambda c, timeout=30: ("boom", 1), 4242) is False
    # an exec that RAISES sent nothing either
    def _boom(cmd, timeout=30):
        raise OSError("daemon gone")
    assert rg.kill_tree(_boom, 4242) is False
    # NO marker is INCONCLUSIVE, and counts as gone: holding a registry row
    # on an unanswerable probe strands the port forever. Said out loud, so
    # nobody reads this as "we assume the kill worked".
    noted = []
    assert rg.kill_tree(lambda c, timeout=30: ("", 0), 4242, log=noted.append) is True
    assert any("no verdict" in m for m in noted), noted


# ── R1 enumeration: neither twin may re-grow its own copy ────────────────────

_FORBIDDEN_LITERALS = ("kill -TERM", "kill -KILL", "kill -0 ", "kill -- -")


def _docstring_ids(tree) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _own_rules(source: str):
    """(kind, lineno) for every place a registry file spells a rule itself
    instead of calling `registry_guard`: a shell kill/probe literal, or a
    comparison of a name called `pid` against 0/1. Docstrings (which
    DESCRIBE the old bug) are not code and are skipped."""
    tree = ast.parse(source)
    docs = _docstring_ids(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            if any(lit in node.value for lit in _FORBIDDEN_LITERALS):
                hits.append(("kill/probe literal", node.lineno))
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str) and any(
                        lit in v.value for lit in _FORBIDDEN_LITERALS):
                    hits.append(("kill/probe literal", node.lineno))
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "pid":
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and comp.value in (0, 1):
                    hits.append(("own pid floor", node.lineno))
    return hits


def _uses_guard(source: str) -> set:
    tree = ast.parse(source)
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "_rg"):
            used.add(node.func.attr)
    return used


@pytest.mark.parametrize("mod", [jobs_mod, services_mod])
def test_the_registry_files_carry_no_private_copy_of_the_rules(mod):
    src = Path(mod.__file__).read_text()
    assert _own_rules(src) == [], _own_rules(src)
    used = _uses_guard(src)
    assert {"pid_state", "kill_tree"} <= used, used
    assert ("valid_pid" in used) or ("validate_row" in used), used


def test_the_enumeration_fires_on_a_private_kill_and_a_private_floor():
    src = Path(services_mod.__file__).read_text()
    assert _own_rules(src) == []
    broken = src + '\n\ndef _stray(pid):\n    if pid <= 1:\n        return\n    cmd = "kill -TERM -- -1"\n'
    kinds = sorted({k for k, _ in _own_rules(broken)})
    assert kinds == ["kill/probe literal", "own pid floor"], kinds


# ── services: the planted row ────────────────────────────────────────────────

def _plant(tmp_path, rows):
    svc = tmp_path / ".services"
    svc.mkdir(parents=True, exist_ok=True)
    (svc / "registry.json").write_text(json.dumps(rows))


@pytest.mark.parametrize("pid", [1, 0, "1", -1, rg.PID_MAX + 1])
def test_a_planted_row_naming_an_unsafe_pid_never_produces_a_kill(tmp_path, pid):
    """Pre-fix: `stop web` ran `kill -TERM -- -1 || kill -TERM 1` — the `--`
    form fails in dash, so the fallback TERMed docker-init: every service,
    every job and tor died. Now the row is dropped at load and nothing is
    signalled."""
    _plant(tmp_path, {"web": {"name": "web", "command": "x", "pid": pid, "port": 8100}})
    sb = FakeSandbox(tmp_path, happy_handler())
    sup = ServiceSupervisor(sb)
    out = sup.stop("web")
    assert not any("sig TERM" in c or "kill -" in c for c in sb.calls), sb.calls
    assert sup._load() == {}
    assert "no service named" in out


def test_a_planted_row_with_a_bad_port_or_name_is_dropped(tmp_path):
    _plant(tmp_path, {"a": {"name": "a", "pid": 4242, "port": 70000},
                      "b": {"name": "../b", "pid": 4242, "port": 8100},
                      "ok": {"name": "ok", "pid": 4242, "port": 8101, "command": "x"}})
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    assert list(sup._load()) == ["ok"]


def test_a_legitimate_stop_uses_the_shared_dash_safe_tree_kill(tmp_path):
    _plant(tmp_path, {"web": {"name": "web", "command": "x", "pid": 4242, "port": 8100}})
    killed = []

    def handler(cmd):
        if "sig TERM" in cmd:
            killed.append(cmd)
            return ("", 0)
        if "kill -0" in cmd:
            return ("", 0 if not killed else 1)
        return ("", 0)
    sb = FakeSandbox(tmp_path, handler)
    sup = ServiceSupervisor(sb)
    assert "stopped" in sup.stop("web")
    assert len(killed) == 1 and "S=4242;" in killed[0] and "-- -" not in killed[0]


def _infra_handler(cmd):
    if "kill -0" in cmd:
        return ("[SANDBOX INFRA ERROR] daemon wedged", 1)
    return ("", 0)


def test_an_infra_fault_does_not_read_as_dead(tmp_path):
    """Pre-fix: the probe's exit 1 read as "dead" — status said DEAD, the
    row's port was a dead claim the allocator re-issued, and a twin could
    be started over a live service."""
    _plant(tmp_path, {"web": {"name": "web", "command": "x", "pid": 4242, "port": 8100}})
    sup = ServiceSupervisor(FakeSandbox(tmp_path, _infra_handler))
    assert sup._entry_state({"pid": 4242}) is None
    status = sup.status()
    assert "UNKNOWN" in status and "DEAD" not in status
    reg = sup._load()
    port, notes = sup._allocate_port(reg, requested=8100)
    assert port != 8100, (port, notes)
    assert "web" in sup._load()                  # the row is kept
    out = sup.start("web", "cmd", port=8100)     # a twin is refused
    assert "already running" in out


def test_a_confirmed_dead_probe_still_reads_as_dead(tmp_path):
    """Control (both worlds agree): a plain exit 1 is death."""
    _plant(tmp_path, {"web": {"name": "web", "command": "x", "pid": 4242, "port": 8100}})

    def dead(cmd):
        return ("", 1) if "kill -0" in cmd else ("", 0)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, dead))
    assert sup._entry_state({"pid": 4242}) is False
    assert "DEAD" in sup.status()


# ── jobs: the nonce survives a restart ───────────────────────────────────────

def _job_sup(root: Path):
    sandbox = MagicMock()
    sandbox.host_workspace = str(root)
    sup = SandboxJobSupervisor(sandbox)
    (root / ".jobs").mkdir(parents=True, exist_ok=True)
    return sup


def test_the_nonce_store_lives_outside_the_bind_mount(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sup = _job_sup(root)
    assert sup._nonce_store is not None
    assert root.resolve() not in sup._nonce_store.resolve().parents
    assert sup._nonce_store.resolve().parent.parent == root.resolve().parent


def test_a_genuine_sentinel_is_accepted_and_a_forged_one_rejected_after_a_restart(tmp_path):
    """Pre-fix: the nonce lived only in memory. A new supervisor over the
    same registry had nonce=None, so `echo 0 > .jobs/<jid>.exit` landed the
    running job as DONE — and the runner's real "<nonce> 7" failed the
    digit check and landed it as LOST."""
    root = tmp_path / "sandbox"
    root.mkdir()
    first = _job_sup(root)
    jid = "job-0000beef"
    first._write_script(jid, "sleep 5", 60.0)         # mints + persists the nonce
    nonce = first._nonces[jid]
    assert 8 <= len(nonce) <= 64
    second = _job_sup(root)                            # "the agent restarted"
    assert second._nonces.get(jid) == nonce
    exit_file = root / ".jobs" / f"{jid}.exit"
    exit_file.write_text("0\n")                        # forged
    assert second._read_exit(jid) is None
    exit_file.write_text(f"{nonce} 7\n")               # genuine
    assert second._read_exit(jid) == 7


def test_a_sentinel_for_a_job_with_no_known_nonce_is_rejected(tmp_path):
    """Fail closed: no nonce known (pre-fix promotion, or a lost store) —
    a bare exit code is exactly the forgery. Pre-fix: accepted as exit 0."""
    root = tmp_path / "sandbox"
    root.mkdir()
    sup = _job_sup(root)
    jid = "job-0000cafe"
    (root / ".jobs" / f"{jid}.exit").write_text("0\n")
    assert sup._read_exit(jid) is None


def test_a_corrupt_or_tampered_nonce_store_trusts_nothing(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sup = _job_sup(root)
    sup._nonce_store.parent.mkdir(parents=True, exist_ok=True)
    sup._nonce_store.write_text("not json")
    assert sup._load_nonces() == {}
    sup._nonce_store.write_text(json.dumps({"job-0000cafe": "short", "../x": "a" * 16,
                                            "job-0000f00d": "a" * 16}))
    assert sup._load_nonces() == {"job-0000f00d": "a" * 16}


def test_jobs_kill_goes_through_the_shared_script(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sup = _job_sup(root)
    sent = []
    sup.sandbox.execute = lambda cmd, timeout=30, **k: sent.append(cmd) or ("", 1)
    sup._kill_pgroup(4242)
    assert sent and "S=4242;" in sent[0] and "sig TERM" in sent[0] and "-- -" not in sent[0]
    sent.clear()
    assert sup._kill_pgroup(1) is False and sent == []


# ── §4GI battery survivors: the two sites the first pins did not reach ───────

def test_a_post_launch_inconclusive_probe_keeps_the_new_row(tmp_path):
    """`start()` probes the pid it just launched; an infra fault there is
    not death. Pre-fix (and the battery's `if _pst is not True: pop`
    mutant) popped the row and reported "exited immediately"."""
    from tests.test_sandbox_services import happy_handler
    happy = happy_handler()

    def handler(cmd):
        return _infra_handler(cmd) if "kill -0" in cmd else happy(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
    out = sup.start("web", "python -m http.server", port=8100)
    assert "exited immediately" not in str(out), out
    assert "web" in sup._load()                        # kept, re-checked later


def test_the_jobs_registry_drops_a_row_above_pid_max_and_keeps_a_valid_one(tmp_path):
    """The jobs `_load` floor goes through `registry_guard.valid_pid`: a pid
    at or above `PID_MAX` is dropped like 0/1 (pre-fix and the `<= 1`
    mutant kept it)."""
    import time as _t
    root = tmp_path / "sandbox"
    sup = _job_sup(root)
    rows = {
        "job-deadbeef": {"pid": rg.PID_MAX + 5, "deadline_at": _t.time() + 600,
                         "state": "running", "cmd": "sleep 1", "started_at": _t.time()},
        "job-0badf00d": {"pid": 4242, "deadline_at": _t.time() + 600,
                         "state": "running", "cmd": "sleep 1", "started_at": _t.time()},
    }
    (root / ".jobs" / "registry.json").write_text(json.dumps(rows))
    reg = sup._load()
    assert "job-0badf00d" in reg                        # the control row survives
    assert "job-deadbeef" not in reg
