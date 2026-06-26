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

from ghost_agent.core.triggers import RecentFailureGuard


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
