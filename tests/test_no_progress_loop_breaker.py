"""In-run no-progress (ungrounded-verification) loop breaker.

Covers the companion to `_note_repeated_failure`: the detector for a turn
loop where every tool call SUCCEEDS but the agent keeps taking the same
action against the same target and getting the same result — the
Browser-OS pathology (double-click the icon → screenshot → "no change" →
repeat) that neither the error-strike counter nor the reasoning-
similarity breaker can see.

These tests pin the decision logic (the module-level helpers) and replay
the exact gating predicate the results loop applies, including the two
exemptions that keep it from firing on legitimate work:
  * mutating calls (iterative file writes) are exempt,
  * errored calls are handled by the failure path, not this one.
"""

from __future__ import annotations

import pytest

from ghost_agent.core.agent import (
    _note_repeated_action,
    _action_result_fingerprint,
)
from ghost_agent.reflection import primary_target_from_args
from ghost_agent.reflection.postmortem import _primary_target
from ghost_agent.distill.schema import ToolCall


# --------------------------------------------------------------------------
# primary_target_from_args — the shared "what did this operate on" key
# --------------------------------------------------------------------------

def test_primary_target_from_args_picks_first_known_key():
    assert primary_target_from_args({"selector": "#FileExplorer"}) == "#fileexplorer"
    assert primary_target_from_args({"path": "Proj/Index.html"}) == "proj/index.html"
    assert primary_target_from_args({"url": "HTTP://x"}) == "http://x"


def test_primary_target_from_args_empty_when_none_present():
    assert primary_target_from_args({"foo": 1}) == ""
    assert primary_target_from_args(None) == ""
    assert primary_target_from_args("not a dict") == ""


def test_primary_target_shared_with_offline_signature():
    # The in-run breaker and the offline post-mortem signature MUST agree
    # on "the same target" — they share one definition.
    args = {"selector": "#term"}
    assert _primary_target(ToolCall(name="browser", arguments=args)) == \
        primary_target_from_args(args)


# --------------------------------------------------------------------------
# _action_result_fingerprint — collapses "same observation", keeps changes
# --------------------------------------------------------------------------

def test_fingerprint_whitespace_insensitive():
    assert _action_result_fingerprint("clicked #x (ok)") == \
        _action_result_fingerprint("clicked   #x  (ok)\n")


def test_fingerprint_distinguishes_real_content_change():
    # A re-read that now returns different bytes (progress) must look
    # different, so it does NOT count as a no-progress repeat.
    assert _action_result_fingerprint("no window open") != \
        _action_result_fingerprint("window opened: FileExplorer")


def test_fingerprint_handles_empty():
    assert _action_result_fingerprint("") == _action_result_fingerprint(None or "")


# --------------------------------------------------------------------------
# _note_repeated_action — trip semantics
# --------------------------------------------------------------------------

def test_note_repeated_action_trips_at_threshold():
    sigs = {}
    fp = _action_result_fingerprint("clicked (ok)")
    counts = [_note_repeated_action(sigs, "browser", "#fe", fp) for _ in range(4)]
    assert [c[1] for c in counts] == [1, 2, 3, 4]
    assert [c[2] for c in counts] == [False, False, True, True]


def test_note_repeated_action_distinct_targets_dont_share_count():
    sigs = {}
    fp = _action_result_fingerprint("clicked (ok)")
    _note_repeated_action(sigs, "browser", "#fe", fp)
    _, cnt, trip = _note_repeated_action(sigs, "browser", "#term", fp)
    assert cnt == 1 and trip is False


def test_note_repeated_action_progress_resets_via_new_fingerprint():
    # Same tool+target, but a CHANGED result is a fresh signature — the
    # no-progress count for the old observation does not carry over.
    sigs = {}
    for _ in range(2):
        _note_repeated_action(sigs, "browser", "#fe", _action_result_fingerprint("no window"))
    _, cnt, trip = _note_repeated_action(
        sigs, "browser", "#fe", _action_result_fingerprint("window opened"))
    assert cnt == 1 and trip is False


def test_custom_threshold():
    sigs = {}
    fp = _action_result_fingerprint("x")
    assert _note_repeated_action(sigs, "t", "", fp, threshold=2)[2] is False
    assert _note_repeated_action(sigs, "t", "", fp, threshold=2)[2] is True


# --------------------------------------------------------------------------
# Scenario replay — the exact gating predicate the results loop applies
# --------------------------------------------------------------------------

def _replay(calls):
    """Replay (fname, target, result, is_mutating, is_error) tuples through
    the SAME predicate the turn loop uses, returning the worst trip seen
    (or None). Mirrors agent.py: only SUCCEEDING, NON-MUTATING calls are
    fed to `_note_repeated_action`."""
    sigs = {}
    worst = None
    for fname, target, result, is_mutating, is_error in calls:
        if fname and not is_mutating and not is_error:
            sig, cnt, trip = _note_repeated_action(
                sigs, fname, target, _action_result_fingerprint(result))
            if trip and (worst is None or cnt > worst[1]):
                worst = (sig, cnt, fname, target)
    return worst


def test_scenario_browser_os_verification_loop_trips():
    # The actual failure: double-click the same icon, screenshot, see no
    # window, repeat — all calls SUCCEED (no error), none mutate.
    calls = []
    for _ in range(4):
        calls.append(("browser", "#fileexplorer", "double-click ok", False, False))
        calls.append(("browser", "", "screenshot saved; desktop visible", False, False))
    trip = _replay(calls)
    assert trip is not None
    assert trip[2] == "browser" and trip[3] == "#fileexplorer"
    assert trip[1] >= 3


def test_scenario_iterative_file_editing_does_not_trip():
    # Legitimate: writing the same file repeatedly while building it up.
    # Writes are MUTATING → exempt → must never look like a no-progress loop.
    calls = [("file_system", "index.html", "wrote 1200 bytes", True, False) for _ in range(6)]
    assert _replay(calls) is None


def test_scenario_repeated_identical_reads_trip():
    # Re-reading the SAME missing/unchanged file over and over IS a
    # no-progress loop (the read-loop pathology) — non-mutating, succeeds
    # with identical content each time.
    calls = [("file_system", "config.json", "{\"a\": 1}", False, False) for _ in range(3)]
    trip = _replay(calls)
    assert trip is not None and trip[3] == "config.json"


def test_scenario_errors_are_not_counted_here():
    # Errored calls are the failure path's job (`_note_repeated_failure`),
    # not this detector — feeding them here must not trip.
    calls = [("file_system", "missing.txt", "Error: not found", False, True) for _ in range(5)]
    assert _replay(calls) is None


def test_scenario_genuine_progress_never_trips():
    # Each call observes something NEW (different result) — healthy work.
    calls = [
        ("browser", "#fe", f"window {i} opened at offset {i}", False, False)
        for i in range(6)
    ]
    assert _replay(calls) is None


# --------------------------------------------------------------------------
# Read/write loop exemption — the FIRST-TRIP REMEDY decision
# --------------------------------------------------------------------------
#
# Tripping the breaker is only half the story; the OTHER half is the remedy.
# For an ordinary re-observation loop the first-trip remedy is to set
# `force_final_response` (drop tools, finish as text). For a tool that both
# READS and WRITES through one dispatch name (manage_composed_skills,
# file_system, ...) that remedy is destructive: the loop is the agent
# re-READING before it performs the WRITE it was asked to do, and forcing a
# text-only turn bars that pending mutation forever. Observed bug: a request
# to reconfigure a composed skill looped on action="list", got force-
# finalised at 3x, and the follow-up action="define" never ran — the change
# silently never landed. For those tools the breaker must STEER but keep
# tools available.

from ghost_agent.core.strikes import (
    is_readwrite_loop_exempt,
    READWRITE_LOOP_TOOLS,
)
from ghost_agent.core.agent import _is_readwrite_loop_exempt


def test_readwrite_exempt_identifies_read_write_tools():
    assert is_readwrite_loop_exempt("manage_composed_skills") is True
    assert is_readwrite_loop_exempt("file_system") is True
    assert is_readwrite_loop_exempt("knowledge_base") is True
    assert is_readwrite_loop_exempt("manage_tasks") is True


def test_readwrite_exempt_excludes_pure_observation_tools():
    # Re-observation tools with no write side must NOT be exempted — their
    # no-progress loop SHOULD force a grounded final answer as before.
    assert is_readwrite_loop_exempt("browser") is False
    assert is_readwrite_loop_exempt("system_weather") is False
    assert is_readwrite_loop_exempt(None) is False
    assert is_readwrite_loop_exempt("") is False


def test_agent_reexports_same_predicate():
    # agent.py imports the helper; the two names must be the same function so
    # the loop breaker and the tests agree.
    assert _is_readwrite_loop_exempt is is_readwrite_loop_exempt


def _first_trip_remedy(fname):
    """Replay the loop breaker's FIRST-TRIP remedy branch (agent.py): returns
    True if `force_final_response` would be set (tools dropped, text-only
    final turn), False if the breaker only steers and KEEPS tools available
    so a pending write can land."""
    return not _is_readwrite_loop_exempt(fname)


def test_remedy_keeps_tools_for_composed_skill_reconfigure():
    # The exact reported bug: looping on a read of manage_composed_skills must
    # NOT force a text-only turn — the pending define/update has to run.
    assert _first_trip_remedy("manage_composed_skills") is False


def test_remedy_forces_final_for_browser_reobservation_loop():
    # The original Browser-OS pathology is unchanged: still force a grounded
    # conclusion, because there is no pending write to protect.
    assert _first_trip_remedy("browser") is True


def test_readwrite_tools_still_trip_the_detector():
    # The exemption changes the REMEDY, not the DETECTION: a repeated read of
    # a read/write tool still trips (so the steer fires) — it just doesn't
    # force a text-only final turn.
    calls = [
        ("manage_composed_skills", "morning_briefing",
         "Composed skills (macros): ...", False, False)
        for _ in range(3)
    ]
    trip = _replay(calls)
    assert trip is not None and trip[2] == "manage_composed_skills"
    # And the remedy for that tripped tool keeps tools available.
    assert _first_trip_remedy(trip[2]) is False
