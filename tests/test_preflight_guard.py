"""Tests for the live pre-flight repeat-failure guard (feature 1A).

Two layers are covered:

* ``RecentFailureGuard`` (``ghost_agent.core.triggers``) — the rolling
  recent-failure memory and its ``would_repeat`` pre-dispatch verdict.
* The wiring on ``GhostAgent`` — the ``--enable-preflight-guard`` flag is
  read into ``_preflight_guard_enabled`` and the guard instance exists.

The full in-loop dispatch interception (the synthetic SYSTEM BLOCK tool
message) lives deep inside ``GhostAgent``'s reasoning loop; its behaviour
is exercised here at the unit boundary it depends on — ``would_repeat``
returning the prior error — which is the contract the loop branches on.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ghost_agent.core.triggers import (
    RecentFailureGuard,
    guard_key_target,
    looks_mutating_command,
)


# ──────────────────────────────────────────────────────────────────────
# RecentFailureGuard — core verdict logic
# ──────────────────────────────────────────────────────────────────────

def test_two_identical_failures_block_the_third():
    """Default (threshold=2): an action may fail the SAME way twice — a
    transient blip or a re-run after an attempted fix — before the third
    identical re-issue is intercepted with the prior error."""
    g = RecentFailureGuard()  # default repeat_threshold=2
    # Nothing recorded yet → allow the first attempt.
    assert g.would_repeat("execute", "x.py") is None
    g.record("execute", "x.py", "Error: SyntaxError: bad token at line 3")
    # One failure on record → still allowed (could be transient / now fixed).
    assert g.would_repeat("execute", "x.py") is None
    g.record("execute", "x.py", "Error: SyntaxError: bad token at line 3")
    # Same error twice → the third attempt is blocked.
    verdict = g.would_repeat("execute", "x.py")
    assert verdict is not None
    assert "syntaxerror" in verdict


def test_threshold_one_blocks_immediate_repeat():
    """An aggressive guard (threshold=1) intercepts the first re-issue."""
    g = RecentFailureGuard(repeat_threshold=1)
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("execute", "x.py") is not None


def test_distinct_target_not_blocked():
    g = RecentFailureGuard(repeat_threshold=1)
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("execute", "y.py") is None


def test_distinct_tool_not_blocked():
    g = RecentFailureGuard(repeat_threshold=1)
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("file_system", "x.py") is None


def test_distinct_error_does_not_trip():
    """The verdict anchors on the MOST RECENT error for the key. Even after
    one error has recurred enough to arm the guard, a NEW failure mode on
    the same target re-anchors and clears the block — modelling 'I changed
    my approach, hit a different wall.'"""
    g = RecentFailureGuard()  # threshold=2
    g.record("execute", "x.py", "Error: SyntaxError at line 3")
    g.record("execute", "x.py", "Error: SyntaxError at line 3")
    # SyntaxError has armed (2 occurrences) — but the latest failure is new:
    g.record("execute", "x.py", "Error: NameError foo undefined")
    # Latest error (NameError) has only one occurrence → not a repeat.
    assert g.would_repeat("execute", "x.py") is None


def test_threshold_two_requires_two_priors():
    """With repeat_threshold=2 the action must already have failed the same
    way twice before the next attempt is blocked (transient-safe mode)."""
    g = RecentFailureGuard(repeat_threshold=2)
    g.record("file_system", "a.txt", "Error: EACCES permission denied")
    # Only one prior identical failure → still allowed.
    assert g.would_repeat("file_system", "a.txt") is None
    g.record("file_system", "a.txt", "Error: EACCES permission denied")
    # Two priors → blocked.
    assert g.would_repeat("file_system", "a.txt") is not None


def test_success_never_seeds_guard():
    """An empty error string is a success and must never arm the guard."""
    g = RecentFailureGuard()
    g.record("execute", "x.py", "")
    assert g.would_repeat("execute", "x.py") is None


def test_empty_tool_is_noop():
    g = RecentFailureGuard()
    g.record("", "x.py", "Error: boom")
    assert g.would_repeat("", "x.py") is None


def test_error_normalisation_collapses_trailing_variance():
    """Two errors that differ only past the 80-char prefix count as the
    same recurring failure."""
    g = RecentFailureGuard()  # threshold=2
    # A >80-char prefix shared by both failures; only the tail (past char 80)
    # differs, so normalisation collapses them to the same recurring failure.
    prefix = "Error: connection refused to 127.0.0.1:8080 while fetching the resource for the user request"
    assert len(prefix) > 80
    g.record("web", "http://t", prefix + " :: pid 111")
    g.record("web", "http://t", prefix + " :: pid 222")
    verdict = g.would_repeat("web", "http://t")
    assert verdict is not None
    assert verdict == prefix[:80].lower()


def test_window_ages_out_old_failures():
    """Failures that scroll out of the bounded window stop blocking."""
    g = RecentFailureGuard(window=4)  # threshold=2
    g.record("execute", "x.py", "Error: boom")
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("execute", "x.py") is not None  # armed (2 in window)
    # Three unrelated failures evict both x.py entries from the size-4 deque.
    g.record("execute", "a.py", "Error: a")
    g.record("execute", "b.py", "Error: b")
    g.record("execute", "c.py", "Error: c")
    assert g.would_repeat("execute", "x.py") is None


def test_reset_clears_history():
    g = RecentFailureGuard()
    g.record("execute", "x.py", "Error: boom")
    g.reset()
    assert g.would_repeat("execute", "x.py") is None


def test_target_is_optional():
    """Tools with no recognised primary target (target == '') still get
    repeat protection keyed on the tool alone."""
    g = RecentFailureGuard()  # threshold=2
    g.record("some_tool", "", "Error: boom")
    g.record("some_tool", "", "Error: boom")
    assert g.would_repeat("some_tool", "") is not None


# ──────────────────────────────────────────────────────────────────────
# World-changed reset + signature keying (2026-07-30 solar-sim postmortem)
#
# Three requests in a row were boxed in by the guard AFTER the model had
# verifiably fixed the failure's cause (killed the process holding the
# port, confirmed it free): blocked calls never dispatch, so the guard
# could never learn the world had moved, and — with no primary-arg key —
# every `manage_services start` (any port, any command) shared one bucket.
# ──────────────────────────────────────────────────────────────────────

def test_world_changed_clears_armed_guard():
    """A successful state mutation between the failures and the retry must
    unblock the retry — the exact live deadlock: two identical port-in-use
    failures, then the port holder is killed, then the (now correct) retry
    was blocked forever."""
    g = RecentFailureGuard()  # threshold=2
    g.record("manage_services", "args#abc", "Error: service 'solar-sim' exited immediately", "start")
    g.record("manage_services", "args#abc", "Error: service 'solar-sim' exited immediately", "start")
    assert g.would_repeat("manage_services", "args#abc", "start") is not None
    cleared = g.note_world_changed()
    assert cleared == 2
    assert g.would_repeat("manage_services", "args#abc", "start") is None


def test_world_changed_returns_zero_when_empty():
    """Zero return lets the call site skip logging no-op resets."""
    g = RecentFailureGuard()
    assert g.note_world_changed() == 0


def test_guard_rearms_after_world_change_if_nothing_was_fixed():
    """The reset is not a pardon: the same failure recurring threshold
    times AFTER the world change re-arms the block."""
    g = RecentFailureGuard()  # threshold=2
    g.record("execute", "x.py", "Error: boom")
    g.record("execute", "x.py", "Error: boom")
    g.note_world_changed()
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("execute", "x.py") is None  # one fresh failure
    g.record("execute", "x.py", "Error: boom")
    assert g.would_repeat("execute", "x.py") is not None  # re-armed


def test_guard_key_target_prefers_primary():
    assert guard_key_target("app.py", 'execute:{"path": "app.py"}') == "app.py"


def test_guard_key_target_signature_distinguishes_changed_args():
    """No primary target → the FULL canonical args string keys the guard,
    so a retry with ANY changed arg (different port, different command) is
    a different action and stays legal."""
    h_8102 = 'manage_services:{"action": "start", "name": "solar-sim", "port": 8102}'
    h_8103 = 'manage_services:{"action": "start", "name": "solar-sim", "port": 8103}'
    k1, k2 = guard_key_target("", h_8102), guard_key_target("", h_8103)
    assert k1.startswith("args#") and k2.startswith("args#")
    assert k1 != k2
    # Deterministic: the identical call maps to the identical key.
    assert guard_key_target("", h_8102) == k1


def test_guard_key_target_empty_everything_stays_empty():
    assert guard_key_target("", "") == ""


def test_identical_reissue_still_blocked_under_signature_keys():
    """The signature fallback must not weaken the guard's core job: a
    byte-identical re-issue of a twice-failed call is still intercepted."""
    g = RecentFailureGuard()  # threshold=2
    key = guard_key_target("", 'manage_services:{"action": "start", "port": 8102}')
    g.record("manage_services", key, "Error: exited immediately", "start")
    g.record("manage_services", key, "Error: exited immediately", "start")
    assert g.would_repeat("manage_services", key, "start") is not None


# ──────────────────────────────────────────────────────────────────────
# looks_mutating_command — world-mutation shell heuristic
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    'kill -9 636 && sleep 1 && echo "Port freed"',        # the live remediation
    "fuser -k 8102/tcp 2>/dev/null; sleep 1",             # ditto (request 92)
    "sleep 1; kill $PID",                                  # verb after separator
    "ps aux | grep solar | awk '{print $2}' | xargs kill -9",
    "bash -c 'kill -9 636'",                               # nested-shell payload is a command
    "rm -rf build/",
    "echo hello > out.txt",                                # bare redirect writes
    "ls -la >> listing.txt",
    "systemctl restart nginx",
    "launchctl bootout system/com.ghost.agent",
    "sed -i 's/a/b/' conf.ini",
    "for f in *.log; do rm $f; done",                      # verb after `do`
])
def test_mutating_commands_detected(cmd):
    assert looks_mutating_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "ss -tlnp | grep 8102 || fuser 8102/tcp 2>/dev/null",  # probe: bare fuser, fd redirect
    "kill -0 674 2>/dev/null && echo alive",               # liveness probe, not a kill
    "pkill -0 solar-sim",                                  # pkill liveness probe
    'curl -s -o /dev/null -w "%{http_code}" http://localhost:8102/',
    "python3 server.py 2>&1 | head",                       # fd dup, not a file write
    "systemctl status nginx",
    "grep -rn services src/",
    "grep -rn 'mkdir' src/",                               # verb as grep PATTERN, not command
    "ls release.tar.gz && shasum foo.tar.xz",              # `tar` inside filenames
    "awk '$3 > 100' data.txt",                              # `>` inside quotes = comparison
    "python3 -c 'print(1 > 0)'",                            # python -c payload is code, not shell
    "",
])
def test_probe_commands_not_mutating(cmd):
    assert looks_mutating_command(cmd) is False


# ──────────────────────────────────────────────────────────────────────
# GhostAgent wiring — flag plumbing
# ──────────────────────────────────────────────────────────────────────

def _make_agent(enable):
    """Construct a GhostAgent with a stub context, bypassing tool
    registry population so __init__ stays cheap."""
    from ghost_agent.core import agent as agent_mod

    ctx = SimpleNamespace(args=SimpleNamespace(enable_preflight_guard=enable))
    with patch.object(agent_mod, "get_available_tools", return_value={}):
        return agent_mod.GhostAgent(ctx)


def test_agent_guard_enabled_by_flag():
    a = _make_agent(True)
    assert a._preflight_guard_enabled is True
    assert isinstance(a._failure_guard, RecentFailureGuard)


def test_agent_guard_disabled_by_flag():
    a = _make_agent(False)
    assert a._preflight_guard_enabled is False
    # The guard object still exists and still records — the flag only
    # governs whether the loop consults it as a hard block.
    assert isinstance(a._failure_guard, RecentFailureGuard)


def test_agent_guard_defaults_on_when_arg_missing():
    """A context whose args lack the attribute (older configs / tests)
    defaults the guard ON."""
    from ghost_agent.core import agent as agent_mod

    ctx = SimpleNamespace(args=SimpleNamespace())  # no enable_preflight_guard
    with patch.object(agent_mod, "get_available_tools", return_value={}):
        a = agent_mod.GhostAgent(ctx)
    assert a._preflight_guard_enabled is True


# ──────────────────────────────────────────────────────────────────────
# CLI flag default
# ──────────────────────────────────────────────────────────────────────

def test_cli_flag_defaults_on():
    from ghost_agent.main import parse_args

    with patch.object(sys, "argv", ["ghost"]):
        args = parse_args()
    assert args.enable_preflight_guard is True


def test_cli_flag_can_be_disabled():
    from ghost_agent.main import parse_args

    with patch.object(sys, "argv", ["ghost", "--no-enable-preflight-guard"]):
        args = parse_args()
    assert args.enable_preflight_guard is False


# ──────────────────────────────────────────────────────────────────────
# Dispatch-loop wiring — lifecycle call sites (source introspection, the
# same boundary style TestBlockBudgetWiring uses: the loop is too deep to
# unit-drive, so assert the contract points exist and agree).
# ──────────────────────────────────────────────────────────────────────

def _agent_src():
    import inspect

    import ghost_agent.core.agent as agent_mod
    return inspect.getsource(agent_mod)


def test_wiring_check_and_record_share_signature_key():
    """Both guard call sites key through guard_key_target, so what the
    check site matches is exactly what the record site wrote."""
    src = _agent_src()
    assert "guard_key_target(_pf_target, a_hash)" in src   # check site
    assert "guard_key_target(ptarget, a_hash)" in src      # record site


def _region(src: str, idx: int) -> str:
    """The branch chain following a marker, bounded by CODE rather than by a
    fixed character count.

    Both pins here used `src[idx:idx + 2600]`, so adding a COMMENT to the
    code they guard pushed the assertion target out of the window and turned
    them red with no behaviour change. Walk to the end of the enclosing
    branch chain instead: the next line at or below the marker's indent that
    is not part of it.
    """
    lines = src[idx:].splitlines(keepends=True)
    base = len(lines[0]) - len(lines[0].lstrip())
    out = [lines[0]]
    for ln in lines[1:]:
        stripped = ln.strip()
        if stripped and not stripped.startswith("#"):
            ind = len(ln) - len(ln.lstrip())
            if ind < base:
                break
        out.append(ln)
        if len("".join(out)) > 8000:
            break
    return "".join(out)


def test_wiring_world_changed_fires_only_on_success():
    """The world-changed reset lives on the SUCCESS branch of the result
    processing (elif of the record-on-failure branch) and is driven by the
    dispatch-time hint."""
    src = _agent_src()
    # The gate gained a third condition on 2026-08-12 (`not _pf_promoted`),
    # so it is now a continuation line — match the stable prefix.
    idx = src.index("elif _pf_world_mut and not _pf_exec_failed")
    assert "note_world_changed()" in _region(src, idx)
    # The hint must never treat blanket-is_mutating `execute` as a world
    # mutation — only heuristic-matched commands (probes stay inert).
    assert "looks_mutating_command(" in src


def test_wiring_failed_execute_never_clears_guard():
    """`execute` failures carry an EXIT CODE banner, not an Error: prefix
    (2026-07-30 review): without exit-code awareness a FAILED remediation
    (`kill` on a stale pid → exit 1) counted as a successful mutation and
    CLEARED the guard. The world-changed branch must gate on the
    execute-aware failure verdict. Deliberately NOT symmetric: execute
    failures are not RECORDED into the guard — repeated identical shell
    failures are the strike ledger's + System-3 pivot's crisis signal, and
    pre-dispatch blocking would starve the pivot (test_system3_crisis_pivot
    caught exactly that when recording was tried)."""
    src = _agent_src()
    idx = src.index("_pf_exec_failed = False")
    region = _region(src, idx)
    assert 'fname == "execute"' in region
    # ⚠ The exit-code regex used to be INLINED here, and this asserted its
    # literal text. It is `ToolOutcome.shell_failed` now — the same reading,
    # shared with the exit-code branch 700 lines below, which had its own
    # copy with a different marker set. Assert the PROPERTY instead: the
    # verdict is exit-code aware, and it comes from the outcome.
    assert "_outcome.shell_failed" in region, (
        "the pre-flight verdict stopped consulting the shell exit code; a "
        "failed remediation (kill on a stale pid -> exit 1) would count as a "
        "successful mutation and CLEAR the guard"
    )
    from ghost_agent.tools.outcome import ToolOutcome
    assert ToolOutcome.ok("done\nEXIT CODE: 1").shell_failed is True
    assert ToolOutcome.ok("done\nEXIT CODE: 0").shell_failed is False
    # Multi-line since 2026-08-12: the reset is additionally gated on
    # `_pf_promoted`, so a DETACHED command — which has not changed the world
    # yet — cannot clear the guard (sandbox/jobs.py).
    assert "elif _pf_world_mut and not _pf_exec_failed" in region
    assert "not _pf_promoted" in region
    # record() stays keyed to genuine FAILURES — never to a refusal.
    # ⚠ Was `assert "if _res_is_error:" in region`. A refusal is now a
    # failure to the loop, and recording one armed the guard against the
    # model's own CORRECTED re-issue: the guard keys on (tool, target, op)
    # and ignores the args, so `replace` without `replace_with` followed by
    # `replace` WITH it is the same key. Measured on the live corpus, a
    # `file_system replace` that really succeeded would have been blocked,
    # under a message telling the model it had re-run something unchanged.
    assert "_res_is_error and not _outcome.is_rejection" in region, (
        "the pre-flight guard records refusals again; the corrected retry "
        "will be blocked"
    )
    assert "not _outcome.changed_the_world" in region, (
        "the guard records a call that DID mutate; its premise is "
        "'re-running this unchanged will fail the same way', which is false "
        "for a half-applied write"
    )


def test_wiring_per_request_reset_exists():
    """A fresh request must not inherit failure memory recorded under a
    previous request's world."""
    src = _agent_src()
    idx = src.index("strikes = StrikeLedger()")
    assert "self._failure_guard.reset()" in src[idx:idx + 1200]
