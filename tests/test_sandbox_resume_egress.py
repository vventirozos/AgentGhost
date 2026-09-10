"""A RESUMED sandbox is a new egress generation (§4FW, 2026-09-10).

§4FU enforced Tor-only egress "once per container generation" and made the
creation path do it. The resume path returns before that code: it starts a
stopped container and reports it ready. `docker start` gives the container a
FRESH network namespace — the GHOST_TOR rules are gone — and starts none of
its processes, so the in-container Tor is not running either.

Measured on the live box, 2026-09-10 11:56, after a restart that resumed the
sandbox rather than recreating it: `iptables -t nat -S` empty, and a plain
`curl https://check.torproject.org/api/ip` from inside the container
answering {"IsTor": false} with the host's real address. Every guarantee
§4FU shipped was off, silently, from the first restart onwards.

The pins drive the real `_try_resume_stopped` through the same exec stub the
§4FU tests use; the last one enumerates the class so the next path that
starts a container cannot skip the enforcement.
"""
import ast
import inspect
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ghost_agent.sandbox import docker as docker_mod
from ghost_agent.sandbox import tor_egress as T
from ghost_agent.sandbox.docker import DockerSandbox

from tests.test_sandbox_tor_egress import _stub, _drive, _cmds

CHAIN_CREATE = "iptables -t nat -N"


def _stopped(tmp_path, **kw):
    """A stub whose container is EXITED and whose readiness probe says 'not
    ready' once (the caller's check) and 'ready' after the start — the real
    sequence `_ensure_running_impl` sees when a stopped container resumes."""
    sb = _stub(tmp_path, **kw)
    sb.container.status = "exited"
    sb._is_container_ready = MagicMock(side_effect=[False, True] + [True] * 20)
    return sb


def test_a_resumed_container_gets_the_rules_and_tor_again(tmp_path):
    """World where it fails: the resume path returns after `mark_ready()`,
    as it did until §4FW — the sandbox comes back with an empty nat table
    and direct egress."""
    sb = _stopped(tmp_path)
    seen, plog = _drive(sb)
    assert sb.container.start.called, "the resume path did not run"
    assert sum(1 for c in _cmds(seen) if CHAIN_CREATE in c) == 1, _cmds(seen)
    assert sb._egress_state == "enforced"


def test_the_resume_re_enforces_even_after_an_enforced_generation(tmp_path):
    """The flag §4FU used says 'already done for this container'. A start is
    a new namespace, so the flag must be RESET by the resume, not honoured."""
    sb = _stub(tmp_path)
    first, _ = _drive(sb)                       # created + enforced
    assert sum(1 for c in _cmds(first) if CHAIN_CREATE in c) == 1
    assert sb._tor_attempted is True
    sb.container.status = "exited"              # the box was restarted
    sb._is_container_ready = MagicMock(side_effect=[False, True] + [True] * 20)
    sb._last_ready_ok = 0.0
    second, _ = _drive(sb)
    assert sum(1 for c in _cmds(second) if CHAIN_CREATE in c) == 1, (
        "the resumed generation kept the previous one's flag")


def test_an_unpaused_container_is_covered_too(tmp_path):
    """Pause/unpause keeps the namespace and the processes, so this is a
    no-op re-apply — but the code must not have to know which of the two it
    is doing."""
    sb = _stopped(tmp_path)
    sb.container.status = "paused"
    seen, _ = _drive(sb)
    assert sb.container.unpause.called
    assert sum(1 for c in _cmds(seen) if CHAIN_CREATE in c) == 1


def test_the_flag_is_what_stops_a_re_apply_not_the_readiness_ttl(tmp_path):
    """Once per generation, proven with the TTL fast path OUT OF THE WAY.

    The §4FU pin drove `ensure_running()` twice and read one application —
    but the second call returned at `_ready_is_fresh()` before reaching the
    enforcement at all, so removing the generation flag left that pin green
    (mutation W3, 2026-09-10). Reset the stamp between the drives and the
    flag is the only thing left holding the line: without it the rules are
    re-applied on every command, through a privileged exec.
    """
    sb = _stub(tmp_path)
    first, _ = _drive(sb)
    assert sum(1 for c in _cmds(first) if CHAIN_CREATE in c) == 1
    sb._last_ready_ok = 0.0          # the TTL cannot mask the second call
    second, _ = _drive(sb)
    assert sum(1 for c in _cmds(second) if CHAIN_CREATE in c) == 0, (
        "the rules were re-applied for a generation already enforced")


def test_without_a_tor_policy_a_resume_enforces_nothing(tmp_path):
    """The discriminating negative: the enforcement is policy-gated, so the
    assertions above are not passing on an unconditional call."""
    sb = _stopped(tmp_path)
    sb.tor_proxy = None
    seen, _ = _drive(sb)
    assert sb.container.start.called
    assert not any("iptables" in c for c in _cmds(seen))


def test_a_resume_that_never_becomes_ready_enforces_nothing(tmp_path):
    """Fail-closed ordering: a container that did not come back is
    recreated by the caller, and the recreation path does the enforcing."""
    sb = _stub(tmp_path)
    sb.container.status = "exited"
    sb._is_container_ready = MagicMock(return_value=False)
    with patch("ghost_agent.sandbox.docker.pretty_log"), \
         patch("ghost_agent.sandbox.docker.time.sleep", lambda *_a: None):
        assert sb._try_resume_stopped() is False
    assert sb._egress_state == ""


# --- the class, not the site ----------------------------------------------

_START_METHODS = {"start", "unpause"}
#: Local names docker.py binds a container handle to. A rename drops the
#: enumeration's count and reds the assertion below rather than silently
#: shrinking what it checks.
_CONTAINER_NAMES = {"c", "container", "old"}


def _starts_a_container(fn) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in _START_METHODS
                and isinstance(f.value, ast.Name)
                and f.value.id in _CONTAINER_NAMES):
            return True
        if (isinstance(f, ast.Attribute) and f.attr in ("run", "create")
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "containers"):
            return True
    return False


def test_every_path_that_starts_a_container_enforces_the_egress():
    """The defect, as a rule: creation enforced and resume did not. Any new
    path that starts or creates the sandbox container must call
    `_enforce_egress_once` in the same function."""
    tree = ast.parse(inspect.getsource(docker_mod))
    starters = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _starts_a_container(fn):
            continue
        starters.append(fn.name)
        calls = {getattr(n.func, "attr", "") for n in ast.walk(fn)
                 if isinstance(n, ast.Call)}
        assert "_enforce_egress_once" in calls, (
            f"{fn.name} starts a container without enforcing the egress")
    assert set(starters) == {"_try_resume_stopped", "_ensure_running_impl"}, (
        f"the start-path enumeration no longer sees both paths: {starters}")


def test_the_enforcement_entry_point_is_the_only_caller_of_the_worker():
    """One implementation: `_enforce_tor_egress` is invoked through the
    guarded entry point, never directly, so the generation flag cannot be
    bypassed."""
    tree = ast.parse(inspect.getsource(docker_mod))
    callers = [fn.name for fn in ast.walk(tree)
               if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
               and fn.name != "_enforce_egress_once"
               and any(getattr(n.func, "attr", "") == "_enforce_tor_egress"
                       for n in ast.walk(fn) if isinstance(n, ast.Call))]
    assert callers == [], callers


def test_the_policy_attribute_is_real_not_only_a_class_default():
    """`tor_proxy` has a class default so a `__new__` stub can reach the
    resume path — but the real constructor must still bind it, or every
    sandbox would silently run with no egress policy."""
    src = inspect.getsource(DockerSandbox.__init__)
    assert "self.tor_proxy" in src and DockerSandbox.tor_proxy is None
