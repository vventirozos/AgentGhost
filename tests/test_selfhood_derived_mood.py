"""Derived mood + staleness prefix (mood rework 2026-08-20).

Covers the two halves of the rework:
  1. ``selfhood/mood.py`` — the pure derivation rules (signals → label,
     ambiguous → None), the age/staleness helpers, and provenance.
  2. The read/write paths that consume them: ``SelfStateThread.set_mood``
     provenance + transition-only history, ``format_as_prefix`` age
     rendering + TTL drop, ``SelfModel.update_derived_mood`` (grace
     window, hysteresis, gating), the ``self_state`` tool's provenance
     tag, and the two agent.py wiring sites (post-turn hook origin gate,
     biological phase 2.8c).

Regression anchor: the live box carried mood="curious" set ONCE on
2026-07-28 and presented it as current for 23 days. Every assertion here
exists so that exact failure shape can't silently return.
"""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.mood import (
    CURIOUS_FRESH_HOURS,
    DERIVED_MOOD_VOCAB,
    IDLE_AFTER_SECONDS,
    MOOD_STALE_AFTER_HOURS,
    SELF_MOOD_GRACE_HOURS,
    MoodSignals,
    age_phrase,
    age_seconds,
    derive_mood,
    mood_is_stale,
    parse_iso_utc,
)
from ghost_agent.selfhood.schema import Experience, _utcnow_iso
from ghost_agent.selfhood.state import SelfStateThread


def _utc(offset_seconds: float = 0.0) -> datetime.datetime:
    return (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=offset_seconds))


def _iso_ago(seconds: float) -> str:
    """On-disk timestamp shape (naive ISO + Z) ``seconds`` in the past."""
    dt = _utc(-seconds).replace(tzinfo=None)
    return dt.isoformat() + "Z"


# ---------------------------------------------------------------------------
# derive_mood — the pure rule table
# ---------------------------------------------------------------------------

def test_lockdown_derives_overloaded_over_everything():
    sig = MoodSignals(
        outcomes=["passed"] * 5, pressure_lockdown=True,
        open_question_count=3, newest_question_age_hours=1.0,
        idle_seconds=9999,
    )
    label, evidence = derive_mood(sig)
    assert label == "overloaded"
    assert "lockdown" in evidence


def test_fail_streak_derives_stuck_with_counts_in_evidence():
    sig = MoodSignals(outcomes=["passed", "failed", "failed", "failed"])
    label, evidence = derive_mood(sig)
    assert label == "stuck"
    assert "3 of my last 4" in evidence


def test_stuck_requires_latest_to_be_a_failure():
    # 3 failures on file but the newest turn passed → not stuck; mixed
    # record → no streak verdict; no questions/idle → None.
    sig = MoodSignals(outcomes=["failed", "failed", "failed", "passed"])
    assert derive_mood(sig) is None


def test_all_pass_streak_derives_satisfied():
    sig = MoodSignals(outcomes=["passed", "passed", "passed", "passed"])
    label, evidence = derive_mood(sig)
    assert label == "satisfied"
    assert "4" in evidence


def test_streak_needs_three_known_verdicts():
    assert derive_mood(MoodSignals(outcomes=["passed", "passed"])) is None
    assert derive_mood(MoodSignals(outcomes=["failed", "failed"])) is None


def test_unknown_and_boot_outcomes_are_ignored():
    sig = MoodSignals(
        outcomes=["boot", "unknown", "passed", "unknown", "passed",
                  "boot", "passed"],
    )
    label, _ = derive_mood(sig)
    assert label == "satisfied"


def test_streak_window_is_newest_five():
    # 5 old failures pushed out of the window by 5 newer passes.
    sig = MoodSignals(outcomes=["failed"] * 5 + ["passed"] * 5)
    label, _ = derive_mood(sig)
    assert label == "satisfied"


def test_fresh_open_question_derives_curious():
    sig = MoodSignals(open_question_count=2, newest_question_age_hours=3.0)
    label, evidence = derive_mood(sig)
    assert label == "curious"
    # Claims only what is verified: the count, and the NEWEST question's
    # freshness (review R5 — "N fresh questions" was falsified by two
    # stale ones). No baked clock reading (review R4).
    assert "2 open questions" in evidence
    assert "newest" in evidence
    assert "ago" not in evidence


def test_mixed_decisive_window_suppresses_curious_but_not_idle():
    """Review R4: with a standing fresh question, alternating verdicts
    flapped stuck↔curious once per turn (history noise). A mixed
    decisive-size window is genuinely ambiguous — curious must not fire
    over it. "idle" is orthogonal (operator absence) and still may."""
    mixed = MoodSignals(
        outcomes=["failed", "passed", "failed", "passed"],
        open_question_count=1, newest_question_age_hours=0.5,
    )
    assert derive_mood(mixed) is None
    mixed_idle = MoodSignals(
        outcomes=["failed", "passed", "failed", "passed"],
        open_question_count=1, newest_question_age_hours=0.5,
        idle_seconds=IDLE_AFTER_SECONDS,
    )
    label, _ = derive_mood(mixed_idle)
    assert label == "idle"


def test_old_open_question_does_not_derive_curious():
    sig = MoodSignals(
        open_question_count=1,
        newest_question_age_hours=CURIOUS_FRESH_HOURS + 1,
    )
    assert derive_mood(sig) is None


def test_curious_boundary_is_inclusive_at_exactly_24h():
    sig = MoodSignals(
        open_question_count=1,
        newest_question_age_hours=CURIOUS_FRESH_HOURS,
    )
    label, _ = derive_mood(sig)
    assert label == "curious"


def test_streak_outranks_curious():
    sig = MoodSignals(
        outcomes=["failed", "failed", "failed"],
        open_question_count=1, newest_question_age_hours=0.5,
    )
    label, _ = derive_mood(sig)
    assert label == "stuck"


def test_idle_derives_idle_only_past_threshold():
    label, evidence = derive_mood(
        MoodSignals(idle_seconds=IDLE_AFTER_SECONDS))
    assert label == "idle"
    assert "background" in evidence
    assert "over 30 minutes" in evidence
    assert derive_mood(
        MoodSignals(idle_seconds=IDLE_AFTER_SECONDS - 1)) is None


def test_idle_evidence_is_clock_free_and_stable():
    """Review R5: a per-reading span ("at least 47m") churned the
    narrative skip-guard sha1 on every heartbeat and baked relative
    time into the diary. Evidence must be identical across idle spans
    and contain no clock reading."""
    _, e1 = derive_mood(MoodSignals(idle_seconds=1900))
    _, e2 = derive_mood(MoodSignals(idle_seconds=3500))
    assert e1 == e2
    assert "ago" not in e1 and "at least" not in e1
    assert derive_mood(
        MoodSignals(idle_seconds=IDLE_AFTER_SECONDS - 1)) is None


def test_no_signals_is_ambiguous_none():
    assert derive_mood(MoodSignals()) is None


def test_every_derivable_label_is_in_the_vocabulary():
    cases = [
        MoodSignals(pressure_lockdown=True),
        MoodSignals(outcomes=["failed"] * 4),
        MoodSignals(outcomes=["passed"] * 4),
        MoodSignals(open_question_count=1, newest_question_age_hours=0.1),
        MoodSignals(idle_seconds=IDLE_AFTER_SECONDS),
    ]
    for sig in cases:
        label, _ = derive_mood(sig)
        assert label in DERIVED_MOOD_VOCAB


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def test_parse_iso_utc_handles_z_aware_and_garbage():
    assert parse_iso_utc(_utcnow_iso()) is not None
    assert parse_iso_utc("2026-08-20T10:00:00+02:00") is not None
    assert parse_iso_utc("") is None
    assert parse_iso_utc("not a timestamp") is None


def test_age_seconds_and_clamp():
    now = _utc()
    assert age_seconds(_iso_ago(3600), now=now) is not None
    assert abs(age_seconds(_iso_ago(3600), now=now) - 3600) < 5
    # Future timestamp (clock skew) clamps to 0, never negative.
    assert age_seconds(_iso_ago(-3600), now=now) == 0.0
    assert age_seconds("garbage", now=now) is None


def test_age_phrase_bands():
    assert age_phrase(10) == "just now"
    assert age_phrase(120) == "2m ago"
    assert age_phrase(7200) == "2h ago"
    assert age_phrase(23 * 86400) == "23d ago"


def test_mood_is_stale_boundaries():
    assert not mood_is_stale(_iso_ago(60))
    # Pin the TTL WIDTH from below too (review R1: a units typo that
    # shrinks the TTL 10× must not survive) — just inside 48h is fresh.
    assert not mood_is_stale(_iso_ago(MOOD_STALE_AFTER_HOURS * 3600 - 60))
    assert mood_is_stale(_iso_ago(MOOD_STALE_AFTER_HOURS * 3600 + 60))
    # Unparseable age is treated as stale — fail-closed toward honesty.
    assert mood_is_stale("garbage")
    assert mood_is_stale("")


def test_parse_iso_utc_tolerates_non_string_garbage():
    """A hand-edited state.json can carry a non-string set_at; the
    helper must return None, not raise (review R1: an AttributeError
    here took down the whole wake-up prefix, not just the mood line)."""
    assert parse_iso_utc(1755000000) is None
    assert parse_iso_utc(None) is None
    assert parse_iso_utc(["2026-08-20"]) is None
    assert mood_is_stale(1755000000)


def test_prefix_survives_non_string_set_at(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("curious", "ev")
    s.note_open_question("does the prefix survive?")
    s._state.mood.set_at = 1755000000  # simulate hand-edited state.json
    s._flush()
    text = SelfStateThread(tmp_path).format_as_prefix()
    assert "curious" not in text            # mood dropped (age unknown)
    assert "does the prefix survive?" in text  # everything else intact


# ---------------------------------------------------------------------------
# SelfStateThread.set_mood — provenance + transition-only history
# ---------------------------------------------------------------------------

def test_set_mood_source_persisted_and_reloaded(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    m = s.set_mood("stuck", "3 fails", source="derived")
    assert m.source == "derived"
    s2 = SelfStateThread(tmp_path)
    assert s2.mood().source == "derived"
    # Default stays "self" (tool path / legacy callers).
    s.set_mood("curious", "reading")
    assert s.mood().source == "self"


def test_set_mood_invalid_source_coerced_to_self(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    m = s.set_mood("curious", source="hallucinated")
    assert m.source == "self"


def test_same_mood_refreshes_without_history_append(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("satisfied", "5 passes", source="derived")
    first_set_at = s.mood().set_at
    assert len(s.mood_history()) == 1

    s.set_mood("satisfied", "6 passes now", source="derived")

    assert len(s.mood_history()) == 1          # heartbeat, not transition
    # Staleness clock reset: _utcnow_iso carries microseconds, so ISO
    # string comparison is chronological and strictly increasing here.
    assert s.mood().set_at > first_set_at
    assert s.mood().evidence == "6 passes now"  # fresh evidence kept
    # And the refresh survives a reload (it was flushed, not just cached).
    s2 = SelfStateThread(tmp_path)
    assert s2.mood().set_at == s.mood().set_at


def test_same_mood_refresh_keeps_prior_evidence_when_new_is_empty(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("satisfied", "5 passes", source="derived")
    s.set_mood("satisfied", "", source="derived")
    assert s.mood().evidence == "5 passes"


def test_label_transition_appends_history_with_source(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("satisfied", "passes", source="derived")
    s.set_mood("stuck", "fails", source="derived")
    hist = s.mood_history()
    assert [m.label for m in hist] == ["satisfied", "stuck"]
    assert all(m.source == "derived" for m in hist)


def test_same_label_different_source_is_a_transition(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("curious", "self says", source="self")
    s.set_mood("curious", "signals say", source="derived")
    assert len(s.mood_history()) == 2
    assert s.mood().source == "derived"


def test_same_label_source_transition_other_direction(tmp_path: Path):
    """derived → self on the same label is ALSO a provenance transition
    (review R1: the dedup must be symmetric — the agent re-authoring the
    current derived label via the tool is a recorded flip, not a
    heartbeat)."""
    s = SelfStateThread(tmp_path)
    s.set_mood("curious", "signals say", source="derived")
    s.set_mood("curious", "I say so myself", source="self")
    hist = s.mood_history()
    assert [m.source for m in hist] == ["derived", "self"]
    assert s.mood().source == "self"


def test_legacy_history_lines_parse_as_self(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.mood_history_path.parent.mkdir(parents=True, exist_ok=True)
    with s.mood_history_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"label": "curious", "evidence": "old",
                            "set_at": "2026-07-28T08:53:59Z"}) + "\n")
    hist = s.mood_history()
    assert hist[0].source == "self"


def test_legacy_state_json_mood_loads_as_self(tmp_path: Path):
    # A pre-rework state.json has no "source" key on the mood record.
    (tmp_path / "state.json").write_text(json.dumps({
        "schema_version": "v1",
        "open_questions": [], "unfinished_threads": [],
        "mood": {"label": "curious", "evidence": "old",
                 "set_at": "2026-07-28T08:53:59Z"},
        "last_session_at": "",
    }), encoding="utf-8")
    s = SelfStateThread(tmp_path)
    assert s.mood().label == "curious"
    assert s.mood().source == "self"


# ---------------------------------------------------------------------------
# format_as_prefix — age rendering + TTL drop
# ---------------------------------------------------------------------------

def test_prefix_shows_fresh_mood_with_provenance_and_age(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("stuck", "3 of 5 failed", source="derived")
    text = s.format_as_prefix()
    assert "stuck" in text
    assert "derived from my recent activity just now" in text
    s.set_mood("curious", "a paper", source="self")
    assert "self-noted just now" in s.format_as_prefix()


def test_prefix_drops_mood_past_ttl(tmp_path: Path):
    """THE regression: a 23-day-old mood must not be presented as
    current. Same state, three clocks — fresh renders, just-inside-TTL
    still renders (pins the TTL width from below), past-TTL doesn't."""
    s = SelfStateThread(tmp_path)
    s.set_mood("curious", "recursive self-reflection")
    s.note_open_question("still relevant?")
    fresh = s.format_as_prefix()
    assert "curious" in fresh
    almost = s.format_as_prefix(now=_utc(MOOD_STALE_AFTER_HOURS * 3600 - 60))
    assert "curious" in almost
    later = _utc(MOOD_STALE_AFTER_HOURS * 3600 + 60)
    stale = s.format_as_prefix(now=later)
    assert "curious" not in stale
    assert "still relevant?" in stale   # the rest of the state survives


def test_prefix_drops_mood_with_unparseable_timestamp(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.set_mood("curious", "ev")
    s._state.mood.set_at = "not-a-timestamp"
    s._flush()
    s2 = SelfStateThread(tmp_path)
    assert "curious" not in s2.format_as_prefix()


# ---------------------------------------------------------------------------
# SelfModel.update_derived_mood — gathering, grace, hysteresis
# ---------------------------------------------------------------------------

def _seed_outcomes(sm: SelfModel, outcomes):
    for i, oc in enumerate(outcomes):
        sm.autobio.append(Experience(
            trajectory_id=f"t-{i}", summary=f"I did task {i}.", outcome=oc,
        ))


def test_update_derives_satisfied_from_pass_streak(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood()
    assert m is not None
    assert m.label == "satisfied"
    assert m.source == "derived"


def test_update_is_a_heartbeat_not_a_history_spam(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    sm.update_derived_mood()
    sm.update_derived_mood()
    sm.update_derived_mood()
    assert len(sm.state.mood_history()) == 1


def test_update_transitions_satisfied_to_stuck(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    sm.update_derived_mood()
    _seed_outcomes(sm, ["failed"] * 4)
    m = sm.update_derived_mood()
    assert m.label == "stuck"
    assert [x.label for x in sm.state.mood_history()] == ["satisfied", "stuck"]


def test_update_ambiguous_writes_nothing(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    assert sm.update_derived_mood() is None
    assert sm.state.mood() is None
    assert sm.state.mood_history() == []


def test_update_respects_fresh_self_mood_grace(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("focused", "deep in code", source="self")
    _seed_outcomes(sm, ["failed"] * 4)
    assert sm.update_derived_mood() is None
    assert sm.state.mood().label == "focused"


def test_update_overrides_self_mood_after_grace(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("focused", "deep in code", source="self")
    _seed_outcomes(sm, ["failed"] * 4)
    later = _utc(SELF_MOOD_GRACE_HOURS * 3600 + 60)
    m = sm.update_derived_mood(now=later)
    assert m is not None and m.label == "stuck"


def test_grace_window_width_holds_just_inside(tmp_path: Path):
    """Pin the grace WIDTH, not just its edges (review R1: a *3600→*60
    units typo collapsed 6h to 6min and every test still passed). A
    self mood aged to one hour BEFORE the grace boundary must still be
    respected against strong contrary signals."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("focused", "deep in code", source="self")
    _seed_outcomes(sm, ["failed"] * 4)
    inside = _utc((SELF_MOOD_GRACE_HOURS - 1.0) * 3600)
    assert sm.update_derived_mood(now=inside) is None
    assert sm.state.mood().label == "focused"


def test_update_overrides_stale_self_mood(tmp_path: Path):
    """The live-box shape: a self mood from weeks ago must not block
    derivation (the grace window only protects a FRESH self mood)."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("curious", "recursive self-reflection", source="self")
    sm.state._state.mood.set_at = _iso_ago(23 * 86400)
    sm.state._flush()
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood()
    assert m is not None and m.label == "satisfied"


def test_update_lockdown_wins_over_streak(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood(pressure_lockdown=True)
    assert m.label == "overloaded"


def test_update_idle_path(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    m = sm.update_derived_mood(idle_seconds=1900.0)
    assert m is not None and m.label == "idle"


def test_update_disabled_selfmodel_is_noop(tmp_path: Path):
    sm = SelfModel(root=tmp_path, enabled=False)
    assert sm.update_derived_mood(idle_seconds=99999) is None


def test_update_fresh_open_question_derives_curious(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.note_open_question("why does the cache invalidate early?")
    m = sm.update_derived_mood()
    assert m is not None
    assert m.label == "curious"
    assert "1 open question" in m.evidence


def test_stats_carry_mood_provenance_and_timestamp(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "fails", source="derived")
    stats = sm.stats()
    assert stats["last_mood"] == "stuck"
    assert stats["last_mood_source"] == "derived"
    assert stats["last_mood_set_at"]


# ---------------------------------------------------------------------------
# self_state tool — provenance + honest listing
# ---------------------------------------------------------------------------

async def test_tool_set_mood_is_source_self(tmp_path: Path):
    from ghost_agent.tools.self_state import tool_self_state
    sm = SelfModel(root=tmp_path)
    out = await tool_self_state(action="set_mood", mood="inquisitive",
                                evidence="new dataset", self_model=sm)
    assert "inquisitive" in out
    assert sm.state.mood().source == "self"


async def test_tool_list_flags_stale_mood(tmp_path: Path):
    from ghost_agent.tools.self_state import tool_self_state
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("curious", "old reflection")
    sm.state._state.mood.set_at = _iso_ago(23 * 86400)
    sm.state._flush()
    listing = await tool_self_state(action="list", self_model=sm)
    assert "curious" in listing        # the tool shows the truth on file…
    assert "STALE" in listing          # …with an explicit flag
    assert "23d ago" in listing


async def test_tool_list_fresh_mood_has_age_no_stale_flag(tmp_path: Path):
    from ghost_agent.tools.self_state import tool_self_state
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "3 fails", source="derived")
    listing = await tool_self_state(action="list", self_model=sm)
    assert "stuck" in listing
    assert "just now" in listing
    assert "STALE" not in listing


async def test_introspect_stats_fresh_mood_shows_age_not_stale(tmp_path: Path):
    """The introspect surface is how users ask "tell me about yourself"
    — its stale flag must not be invertible undetected (review R1)."""
    from ghost_agent.tools.introspect import tool_introspect
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("satisfied", "streak", source="derived")
    out = await tool_introspect(action="stats", self_model=sm)
    assert "satisfied" in out
    assert "derived from my recent activity just now" in out
    assert "STALE" not in out


async def test_introspect_stats_stale_mood_flagged(tmp_path: Path):
    from ghost_agent.tools.introspect import tool_introspect
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("curious", "old reflection")
    sm.state._state.mood.set_at = _iso_ago(23 * 86400)
    sm.state._flush()
    out = await tool_introspect(action="stats", self_model=sm)
    assert "curious" in out
    assert "STALE" in out
    assert "23d ago" in out


# ---------------------------------------------------------------------------
# Acute-state retirement (review R2: falsified readings must not stand)
# ---------------------------------------------------------------------------

def test_idle_mood_retired_by_next_user_turn(tmp_path: Path):
    """The live-shaped bug: phase 2.8c writes "idle", the operator
    returns and chats, signals are ambiguous — the idle label must be
    CLEARED, not presented as current for 48h."""
    sm = SelfModel(root=tmp_path)
    m = sm.update_derived_mood(idle_seconds=1900.0)
    assert m.label == "idle"
    # Post-turn update from a REAL operator turn, no signals (ambiguous):
    assert sm.update_derived_mood(operator_turn=True) is None
    assert sm.state.mood() is None


def test_idle_mood_not_retired_by_internal_turn(tmp_path: Path):
    """Review R4: scheduled tasks / job resumes run the same post-turn
    hook at 3am with the operator genuinely away — they must not retire
    "idle" (the hook passes operator_turn=False for sched-/job-/sub-
    request ids)."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(idle_seconds=1900.0)
    assert sm.update_derived_mood(operator_turn=False) is None
    assert sm.state.mood() is not None
    assert sm.state.mood().label == "idle"
    assert [x.label for x in sm.state.mood_history()] == ["idle"]


def test_idle_mood_holds_against_internal_turn_streak(tmp_path: Path):
    """Acute-hold: an internal 3am turn with a decisive pass streak may
    not replace a standing "idle" — the operator is still away, and the
    gauge is operator-facing. An OPERATOR turn with the same streak
    replaces it."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(idle_seconds=1900.0)
    _seed_outcomes(sm, ["passed"] * 4)
    assert sm.update_derived_mood(operator_turn=False) is None
    assert sm.state.mood().label == "idle"
    m = sm.update_derived_mood(operator_turn=True)
    assert m is not None and m.label == "satisfied"


def test_idle_mood_yields_to_fresh_overloaded_from_any_turn(tmp_path: Path):
    """Acute-over-acute: a turn that genuinely locked down (even an
    internal one) carries a fresher acute reading than a standing
    "idle" and wins."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(idle_seconds=1900.0)
    m = sm.update_derived_mood(pressure_lockdown=True, operator_turn=False)
    assert m is not None and m.label == "overloaded"


def test_overloaded_retired_by_next_clean_turn(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    m = sm.update_derived_mood(pressure_lockdown=True)
    assert m.label == "overloaded"
    assert sm.update_derived_mood(pressure_lockdown=False) is None
    assert sm.state.mood() is None


def test_overloaded_not_retired_by_idle_tick(tmp_path: Path):
    """An idle tick brings no new turn — "the last turn ended under
    lockdown" is still true, so the reading stands (only clock-based
    aging applies). idle_seconds below the idle threshold keeps the
    derivation ambiguous."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(pressure_lockdown=True)
    assert sm.update_derived_mood(idle_seconds=1000.0) is None
    assert sm.state.mood() is not None
    assert sm.state.mood().label == "overloaded"


def test_overloaded_holds_against_idle_tick_streak(tmp_path: Path):
    """Review R4 (stress-walk MAJOR): the idle tick is lockdown-blind —
    the same pass streak that was OUTRANKED at mint time must not win
    40 minutes later just because this caller can't see the flag. Only
    a completed turn may replace "overloaded"."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood(pressure_lockdown=True, operator_turn=True)
    assert m.label == "overloaded"
    # Idle tick, streak decisive → held.
    assert sm.update_derived_mood(idle_seconds=1000.0) is None
    assert sm.state.mood().label == "overloaded"
    # Idle tick deriving "idle" → also held (no falsifying turn).
    assert sm.update_derived_mood(idle_seconds=1900.0) is None
    assert sm.state.mood().label == "overloaded"
    # The next completed clean turn IS entitled: streak replaces it.
    m2 = sm.update_derived_mood(operator_turn=True)
    assert m2 is not None and m2.label == "satisfied"


def test_self_moods_are_never_retired(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("idle", "I feel dormant", source="self")
    # With no signals the derivation is ambiguous; what protects the
    # mood here is retirement's source-guard (self moods are excluded
    # from retirement), not the grace window — grace only guards
    # WRITES, and no write is attempted on the ambiguous path.
    assert sm.update_derived_mood() is None
    assert sm.state.mood() is not None
    assert sm.state.mood().label == "idle"


def test_idle_mood_not_retired_by_sawtoothed_idle_tick(tmp_path: Path):
    """Review R3 MAJOR: phase 3's self-play finally resets
    last_activity_time, saw-toothing the idle clock while the operator
    is genuinely AFK. An idle tick with a small idle reading therefore
    does NOT prove the operator returned — it must never retire a
    standing "idle" mood (only a completed user turn, idle_seconds==0,
    may). Retiring here produced a probe-proven retire→re-derive flap
    with duplicate history lines and log churn."""
    sm = SelfModel(root=tmp_path)
    m = sm.update_derived_mood(idle_seconds=1900.0)
    assert m.label == "idle"
    assert sm.update_derived_mood(idle_seconds=1000.0) is None
    assert sm.state.mood() is not None
    assert sm.state.mood().label == "idle"
    # The whole AFK stretch produced exactly ONE history line.
    assert [x.label for x in sm.state.mood_history()] == ["idle"]


def test_retirement_helper_honours_lockdown_contract(tmp_path: Path):
    """The lockdown guard in _maybe_retire_acute_mood is unreachable
    from the current call site (derive_mood returns "overloaded"
    whenever the flag is set) but is the helper's own contract — pin it
    directly so it is an executed guard, not a token one."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(pressure_lockdown=True)
    assert sm.state.mood().label == "overloaded"
    assert sm._maybe_retire_acute_mood(
        pressure_lockdown=True, idle_seconds=0.0) is False
    assert sm.state.mood() is not None
    assert sm._maybe_retire_acute_mood(
        pressure_lockdown=False, idle_seconds=0.0) is True
    assert sm.state.mood() is None


def test_internal_clean_turn_retires_overloaded_not_replaces(tmp_path: Path):
    """Review R5 (laundering): idle→overloaded→satisfied via two
    internal turns displaced "idle" with the operator away all night —
    each step individually entitled. Fix: an internal clean turn
    FALSIFIES "overloaded" (retires it) but must not mint the
    replacement standing label; only an operator turn may."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood(pressure_lockdown=True, operator_turn=True)
    assert m.label == "overloaded"
    out = sm.update_derived_mood(operator_turn=False)   # internal clean turn
    assert out is None
    assert sm.state.mood() is None                      # retired, not replaced
    # History records only states held — no "satisfied" was minted.
    assert [x.label for x in sm.state.mood_history()][-1] == "overloaded"


def test_stale_acute_prior_does_not_hold_the_gauge_dark(tmp_path: Path):
    """Review R5: a held acute mood that has gone stale (>48h) is
    already dropped/flagged on every surface — it must not keep vetoing
    a live decisive derivation from the tick."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(pressure_lockdown=True, operator_turn=True)
    sm.state._state.mood.set_at = _iso_ago(49 * 3600)
    sm.state._flush()
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood(idle_seconds=1000.0)     # lockdown-blind tick
    assert m is not None and m.label == "satisfied"


def test_heartbeats_keep_narrative_state_block_byte_stable(tmp_path: Path):
    """Review R5: the narrative skip-guard sha1s its prompt — idle
    heartbeats at different clock readings must render an identical
    age-free state block (evidence is clock-free by construction)."""
    sm = SelfModel(root=tmp_path)
    sm.update_derived_mood(idle_seconds=1900.0)
    a = sm.narrative._format_state_block(sm.state)
    sm.update_derived_mood(idle_seconds=3400.0)         # later heartbeat
    b = sm.narrative._format_state_block(sm.state)
    assert a == b


def test_mood_arc_excludes_stale_history(tmp_path: Path):
    """Review R5: "Mood arc lately" rendered a 23-day-old label into
    the diary prompt as recent. The arc shares the 48h TTL."""
    sm = SelfModel(root=tmp_path)
    s = sm.state
    s.set_mood("stuck", "old", source="derived")
    s.set_mood("satisfied", "old too", source="self")
    # Age both history lines and the slot far past the TTL.
    import json as _json
    lines = []
    for ln in s.mood_history_path.read_text(encoding="utf-8").splitlines():
        d = _json.loads(ln)
        d["set_at"] = _iso_ago(23 * 86400)
        lines.append(_json.dumps(d))
    s.mood_history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    s._state.mood.set_at = _iso_ago(23 * 86400)
    s._flush()
    block = sm.narrative._format_state_block(s)
    assert "Mood arc lately" not in block
    # Fresh transitions still render an arc.
    s.set_mood("curious", "fresh", source="derived")
    s.set_mood("stuck", "fresher", source="derived")
    assert "Mood arc lately" in sm.narrative._format_state_block(s)


def test_mood_arc_keeps_exactly_one_stale_predecessor(tmp_path: Path):
    """Review R7: an off-by-two slice (fresh_idx[0]-2) survived the
    suite — it re-admits stale labels that are NOT the from-state into
    the "lately" arc (the R5 defect class). Pin the identity: exactly
    one predecessor, nothing before it."""
    import json as _json
    sm = SelfModel(root=tmp_path)
    s = sm.state
    s.set_mood("curious", "very old", source="derived")
    s.set_mood("stuck", "old hold", source="derived")
    s.set_mood("satisfied", "fresh shift", source="derived")
    lines = s.mood_history_path.read_text(encoding="utf-8").splitlines()
    aged = []
    for i, ln in enumerate(lines):
        d = _json.loads(ln)
        if i < 2:                       # curious + stuck entries: stale
            d["set_at"] = _iso_ago((30 - i) * 86400)
        aged.append(_json.dumps(d))
    s.mood_history_path.write_text("\n".join(aged) + "\n", encoding="utf-8")
    block = sm.narrative._format_state_block(s)
    assert "Mood arc lately: stuck → satisfied." in block
    assert "curious" not in block       # NOT the from-state — excluded


def test_narrative_block_carries_provenance_wording(tmp_path: Path):
    """Review R7: a provenance-drop mutant of the no-age rendering
    survived — docs promise "label + provenance only", so pin both
    halves of the wording via the shared helper's phrases."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "ev", source="derived")
    assert ("My mood, derived from my recent activity: stuck."
            in sm.narrative._format_state_block(sm.state))
    sm.state.set_mood("focused", "ev", source="self")
    assert ("My mood, self-noted: focused."
            in sm.narrative._format_state_block(sm.state))


def test_recall_query_uses_label_not_evidence(tmp_path: Path):
    """Review R7: evidence in the recall query made retrieval depend on
    clock-varying tokens (masked only by the tokenizer's ≤2-char drop).
    The query must carry the label, never the evidence."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "EVIDENCE_SENTINEL_QRS keeps failing",
                      source="derived")
    q = sm.narrative._derive_recall_query(sm.state)
    assert "stuck" in q
    assert "EVIDENCE_SENTINEL_QRS" not in q


def test_standing_states_are_not_retired_by_ambiguity(tmp_path: Path):
    """stuck/satisfied/curious are standing assessments — ambiguous
    signals leave them in place (they age out via the TTL)."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["failed"] * 4)
    m = sm.update_derived_mood()
    assert m.label == "stuck"
    # Wipe the log so the next update sees no verdicts (ambiguous).
    sm.autobio.path.unlink()
    assert sm.update_derived_mood() is None
    assert sm.state.mood().label == "stuck"


# ---------------------------------------------------------------------------
# Verdict back-scan (review R2: raw-tail read starved the streak)
# ---------------------------------------------------------------------------

def test_recent_verdicts_sees_through_boot_and_unknown_tail(tmp_path: Path):
    """The live tail shape: verdicts buried under a burst of boot
    markers and unknown-verdict turns. A raw recent(16) read yields
    zero verdicts; the back-scan must still find them."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    for i in range(20):
        sm.autobio.append(Experience(
            trajectory_id=f"u-{i}", summary=f"I chatted about topic {i}.",
            outcome="unknown"))
    for _ in range(8):
        sm.autobio._raw_append(Experience(
            summary="Session resumed.", outcome="boot", cluster="meta"))
    assert sm.autobio.recent_verdicts(limit=5) == ["passed"] * 4
    m = sm.update_derived_mood()
    assert m is not None and m.label == "satisfied"


def test_recent_verdicts_order_and_limit(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["failed", "passed", "failed", "failed", "passed",
                        "failed"])
    v = sm.autobio.recent_verdicts(limit=5)
    assert v == ["passed", "failed", "failed", "passed", "failed"]


def test_recent_verdicts_age_bound(tmp_path: Path):
    """Review R3 MAJOR: without an age bound, weeks-old verdicts
    reached by the back-scan keep minting a "fresh" streak mood via the
    heartbeat set_at refresh — the eternal-label defect reborn."""
    sm = SelfModel(root=tmp_path)
    for i in range(4):
        sm.autobio.append(Experience(
            trajectory_id=f"old-{i}", summary=f"I did old task {i}.",
            outcome="passed", timestamp=_iso_ago(21 * 86400)))
    assert sm.autobio.recent_verdicts(limit=5, max_age_days=7.0) == []
    # Fresh verdicts inside the bound still count.
    _seed_outcomes(sm, ["failed", "failed"])
    assert sm.autobio.recent_verdicts(
        limit=5, max_age_days=7.0) == ["failed", "failed"]
    # No bound → all visible (back-compat of the raw scan).
    assert len(sm.autobio.recent_verdicts(limit=6)) == 6


def test_streak_age_bound_admits_recent_verdicts(tmp_path: Path):
    """Pin the age-bound WIDTH from below (review R4: a
    STREAK_MAX_AGE_DAYS=1.0 mutant survived — at live verdict rates a
    1-day bound admits <3 verdicts and silently re-creates the R2
    starvation). Three-day-old verdicts are well inside the 7-day bound
    and must still derive."""
    sm = SelfModel(root=tmp_path)
    for i in range(4):
        sm.autobio.append(Experience(
            trajectory_id=f"r-{i}", summary=f"I did recent task {i}.",
            outcome="passed", timestamp=_iso_ago(3 * 86400)))
    m = sm.update_derived_mood()
    assert m is not None and m.label == "satisfied"


def test_recent_verdicts_drops_garbage_timestamps_when_bounded(tmp_path: Path):
    """A record whose timestamp doesn't parse has an unknowable age —
    when an age bound is in force it must be DROPPED (counting it as
    fresh is the direction that re-opens the eternal-streak defect via
    corrupt records — review R4)."""
    sm = SelfModel(root=tmp_path)
    sm.autobio.append(Experience(
        trajectory_id="g", summary="I did a corrupt-stamped task.",
        outcome="passed", timestamp="not-a-time"))
    _seed_outcomes(sm, ["passed"])
    assert sm.autobio.recent_verdicts(limit=5, max_age_days=7.0) == ["passed"]
    # Unbounded scans keep it (age is only enforced with a bound).
    assert len(sm.autobio.recent_verdicts(limit=5)) == 2


def test_stale_verdicts_cannot_keep_a_mood_eternally_fresh(tmp_path: Path):
    """End-to-end pin of the R3 probe: satisfied derived today, then
    zero activity for 21 days — the mood must NOT be re-refreshed off
    the old verdicts, and the prefix must have aged it out."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 4)
    m = sm.update_derived_mood()
    assert m is not None and m.label == "satisfied"
    later = _utc(21 * 86400)
    assert sm.update_derived_mood(now=later) is None   # verdicts aged out
    assert "satisfied" not in sm.state.format_as_prefix(now=later)


def test_recent_verdicts_bounded_scan(tmp_path: Path):
    """Verdicts beyond the scan window are (deliberately) not found —
    the scan is bounded, not a full-file parse."""
    sm = SelfModel(root=tmp_path)
    _seed_outcomes(sm, ["passed"] * 2)
    for i in range(250):
        sm.autobio.append(Experience(
            trajectory_id=f"u-{i}", summary=f"I chatted about topic {i}.",
            outcome="unknown"))
    assert sm.autobio.recent_verdicts(limit=5, scan_limit=200) == []


# ---------------------------------------------------------------------------
# Input caps + age-free narrative rendering (review R2)
# ---------------------------------------------------------------------------

def test_set_mood_caps_label_and_evidence(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    m = s.set_mood("x" * 500, "e" * 5000)
    assert len(m.label) <= 64
    assert len(m.evidence) <= 300


def test_prefix_without_age_is_clock_stable(tmp_path: Path):
    """The narrative consumer renders the mood WITHOUT the age phrase —
    the prompt sha1 is the regenerate skip-guard, and a wall-clock-
    varying '2h ago' would defeat it."""
    s = SelfStateThread(tmp_path)
    s.set_mood("stuck", "3 fails", source="derived")
    a = s.format_as_prefix(include_mood_age=False)
    b = s.format_as_prefix(include_mood_age=False,
                           now=_utc(3600.0))   # an hour later, same state
    assert a == b
    assert "ago" not in a and "just now" not in a
    assert "stuck" in a
    # Staleness drop still applies even without the age phrase.
    c = s.format_as_prefix(include_mood_age=False,
                           now=_utc(MOOD_STALE_AFTER_HOURS * 3600 + 60))
    assert "stuck" not in c


def test_narrative_state_block_has_no_age_phrase(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "3 fails", source="derived")
    block = sm.narrative._format_state_block(sm.state)
    assert "stuck" in block
    assert "ago" not in block and "just now" not in block


def test_narrative_state_block_carries_no_evidence(tmp_path: Path):
    """Review R6: evidence strings legitimately vary with the clock
    (streak counts drain as verdicts cross the 7-day bound with zero
    new events) — the narrative rendering must be stable by
    CONSTRUCTION: label + provenance only."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "EVIDENCE_SENTINEL_XYZ", source="derived")
    block = sm.narrative._format_state_block(sm.state)
    assert "stuck" in block
    assert "EVIDENCE_SENTINEL_XYZ" not in block


def test_narrative_block_stable_across_verdict_age_drain(tmp_path: Path):
    """Review R6 (probe-proven): two verdicts crossing the 7-day age
    bound between heartbeats changed the slot evidence ("my last 5" →
    "my last 3") with ZERO new events — the narrative block must not
    change bytes over it."""
    sm = SelfModel(root=tmp_path)
    for i in range(5):
        age_days = 6.9 if i < 2 else 0.1
        sm.autobio.append(Experience(
            trajectory_id=f"d-{i}", summary=f"I did drain task {i}.",
            outcome="passed", timestamp=_iso_ago(age_days * 86400)))
    m = sm.update_derived_mood(operator_turn=True)
    assert m.label == "satisfied" and "5" in m.evidence
    a = sm.narrative._format_state_block(sm.state)
    later = _utc(0.2 * 86400)   # the two old verdicts cross the bound
    m2 = sm.update_derived_mood(idle_seconds=1000.0, now=later)
    assert m2 is not None and "3" in m2.evidence   # slot evidence drained
    b = sm.narrative._format_state_block(sm.state)
    assert a == b   # diary prompt unchanged — no spurious regeneration


def test_mood_arc_includes_from_state_of_fresh_transition(tmp_path: Path):
    """Review R6: a state ENTERED 3 days ago but held until a
    transition 1h ago is exactly the "lately" arc — filtering each
    entry by its own set_at dropped the from-state and left the arc
    dark at live rates."""
    import json as _json
    sm = SelfModel(root=tmp_path)
    s = sm.state
    s.set_mood("stuck", "entered long ago", source="derived")
    s.set_mood("satisfied", "fresh shift", source="derived")
    lines = s.mood_history_path.read_text(encoding="utf-8").splitlines()
    d0 = _json.loads(lines[0])
    d0["set_at"] = _iso_ago(3 * 86400)   # age the stuck ENTRY 3 days
    s.mood_history_path.write_text(
        _json.dumps(d0) + "\n" + lines[1] + "\n", encoding="utf-8")
    block = sm.narrative._format_state_block(s)
    assert "Mood arc lately: stuck → satisfied." in block


# ---------------------------------------------------------------------------
# Narrative recall query — stale mood must not steer diary recall
# ---------------------------------------------------------------------------

def test_recall_query_includes_fresh_mood(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "parser keeps failing", source="derived")
    q = sm.narrative._derive_recall_query(sm.state)
    assert "stuck" in q


def test_recall_query_excludes_stale_mood(tmp_path: Path):
    """Review R1: _derive_recall_query was the one surviving stale-mood
    leak — a 23-day-old label kept biasing IDF diary recall."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("curious", "recursive self-reflection")
    sm.state._state.mood.set_at = _iso_ago(23 * 86400)
    sm.state._flush()
    sm.state.note_open_question("what changed in the cache layer?")
    q = sm.narrative._derive_recall_query(sm.state)
    assert "curious" not in q
    assert "recursive self-reflection" not in q
    assert "cache layer" in q   # open questions still contribute


# ---------------------------------------------------------------------------
# Wiring: post-turn hook in _record_turn_trajectory
# ---------------------------------------------------------------------------

def _make_traj_agent(tmp_path: Path):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.trajectory_collector = MagicMock()
    agent.context.trajectory_collector.append = MagicMock(return_value=None)
    # A MagicMock attr is truthy — pin the pressure flag to a real bool
    # (production resets it at request start).
    agent.context._ctx_pressure_lockdown = False
    # turn_origin: no explicit label (must be absent, not a MagicMock
    # string) + a skill store that is not read-only → "user".
    agent.context.turn_origin_label = None
    agent.context.skill_memory = SimpleNamespace(is_read_only=False)
    agent.context.self_model = SelfModel(root=tmp_path)
    return agent


def test_post_turn_hook_calls_update_derived_mood(tmp_path: Path):
    from ghost_agent.core.agent import GhostAgent
    agent = _make_traj_agent(tmp_path)
    calls = []
    agent.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "do the thing"}],
        final_content="done", req_id="r1", model="m",
        user_request="do the thing",
    )
    assert len(calls) == 1
    assert calls[0]["pressure_lockdown"] is False
    assert calls[0]["operator_turn"] is True   # plain req id = operator


def test_post_turn_hook_internal_request_is_not_operator_turn(tmp_path: Path):
    """Review R4: sched-/job-/sub- request ids are autonomous turns —
    the hook must pass operator_turn=False so a 3am scheduled task
    cannot retire a genuine "idle"."""
    from ghost_agent.core.agent import GhostAgent
    agent = _make_traj_agent(tmp_path)
    calls = []
    agent.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "nightly routine"}],
        final_content="done", req_id="sched-abc123", model="m",
        user_request="nightly routine",
    )
    assert calls and calls[0]["operator_turn"] is False


def test_post_turn_hook_passes_lockdown_flag(tmp_path: Path):
    from ghost_agent.core.agent import GhostAgent
    agent = _make_traj_agent(tmp_path)
    agent.context._ctx_pressure_lockdown = True
    calls = []
    agent.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "do the thing"}],
        final_content="done", req_id="r2", model="m",
        user_request="do the thing",
    )
    assert calls and calls[0]["pressure_lockdown"] is True


def test_post_turn_hook_prefers_snapshot_over_live_flag(tmp_path: Path):
    """Review R1 MAJOR: the SSE drain runs post-semaphore, where the
    live _ctx_pressure_lockdown belongs to whichever request runs NOW.
    An explicit pressure_lockdown snapshot must win over the live flag
    — in both directions."""
    from ghost_agent.core.agent import GhostAgent
    agent = _make_traj_agent(tmp_path)
    calls = []
    agent.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    # Turn ended under lockdown; a concurrent request has since reset
    # the live flag to False. The snapshot (True) must win.
    agent.context._ctx_pressure_lockdown = False
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "do the thing"}],
        final_content="done", req_id="r4", model="m",
        user_request="do the thing", pressure_lockdown=True,
    )
    # Inverse: turn did NOT lock down; a concurrent request has since
    # armed the live flag. The snapshot (False) must win.
    agent.context._ctx_pressure_lockdown = True
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "do the thing"}],
        final_content="done", req_id="r5", model="m",
        user_request="do the thing", pressure_lockdown=False,
    )
    assert [c["pressure_lockdown"] for c in calls] == [True, False]


async def test_stream_drain_uses_snapshot_not_live_flag(tmp_path: Path):
    """Executed end-to-end pin for the R1 MAJOR: drive the REAL SSE
    drain (_stream_final_generation) with a StreamState whose
    semaphore-time snapshot says False while the LIVE context flag says
    True (as if a concurrent request armed it post-semaphore). The
    derived-mood hook must see the snapshot, not the live flag."""
    import json as _json
    from tests.test_finalize_stream_pins import (
        make_stream_agent, _make_stream_state)

    a = make_stream_agent()
    a.context.self_model = SelfModel(root=tmp_path)
    calls = []
    a.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    a.context.trajectory_collector = MagicMock()
    a.context.trajectory_collector.append = MagicMock(return_value=None)
    a.context.skill_memory = SimpleNamespace(is_read_only=False)
    a.context.turn_origin_label = None
    # The live flag belongs to "whichever request runs now" — arm it.
    a.context._ctx_pressure_lockdown = True

    async def final_stream(payload, use_coding=False):
        yield (f"data: {_json.dumps({'choices': [{'delta': {'content': 'A fine answer.'}}]})}\n\n"
               .encode())
        yield b"data: [DONE]\n\n"

    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    ss = _make_stream_state(reg)   # pressure_lockdown defaults False
    gen, _, _ = a._stream_final_generation(ss)
    async for _ in gen:
        pass

    assert calls, "derived-mood hook never fired on the streamed drain"
    assert calls[0]["pressure_lockdown"] is False


async def test_full_streamed_turn_snapshots_lockdown_at_build_site(tmp_path: Path):
    """Executed pin for the BUILD-SITE half of the snapshot identity
    (review R2 MAJOR: the drain half was pinned, but deleting the one
    kwarg at the StreamState build site silently reverts to the
    default-False field and the suite stayed green). Drive the REAL
    handle_chat with stream:true; arm the lockdown flag mid-turn (after
    the request-start reset, as the context governor's second overflow
    would); then disarm the LIVE flag before draining. The hook must
    still see True — proof the value was captured at build time."""
    import json as _json
    from unittest.mock import patch
    from tests.test_finalize_stream_pins import make_stream_agent

    a = make_stream_agent()
    a.context.self_model = SelfModel(root=tmp_path)
    calls = []
    a.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    a.context.trajectory_collector = MagicMock()
    a.context.trajectory_collector.append = MagicMock(return_value=None)
    a.context.skill_memory = SimpleNamespace(is_read_only=False)
    a.context.turn_origin_label = None

    def _sse(delta):
        return (f"data: {_json.dumps({'choices': [{'delta': delta}]})}\n\n"
                .encode())

    # Two-act script: response 1 emits a tool_call while the reasoning
    # channel disclaims tools — the dual-channel drop at the "reasoning
    # channel disclaimed tools" guard sets force_final_response and
    # loops; response 2 is then a FORCED-FINAL generation, which with
    # stream:true takes the client-SSE branch that builds StreamState
    # (the build site under test). Verified against the live guard.
    call_n = {"n": 0}

    def make_stream(*a_, **k_):
        call_n["n"] += 1
        n = call_n["n"]

        async def gen():
            if n == 1:
                yield _sse({"reasoning_content":
                            "I can answer this directly, no tools needed."})
                yield _sse({"content":
                            "<tool_call>\n{\"name\": \"execute\", "
                            "\"arguments\": {\"code\": \"1\"}}\n"
                            "</tool_call>"})
            else:
                yield _sse({"content": "A plain streamed final answer."})
            yield b"data: [DONE]\n\n"
        return gen()

    a.context.llm_client.stream_chat_completion = make_stream

    def arm_and_no_tools(*a_, **k_):
        # Runs mid-turn, after handle_chat's request-start reset of the
        # flag — stands in for the governor's second-overflow arming.
        a.context._ctx_pressure_lockdown = True
        return []

    body = {"messages": [{"role": "user", "content": "stream an answer"}],
            "stream": True}
    with patch("ghost_agent.core.agent.get_active_tool_definitions",
               side_effect=arm_and_no_tools):
        out = await a.handle_chat(
            body, background_tasks=MagicMock(), request_id="req-mood-snap")

    # Simulate a concurrent request resetting the LIVE flag before the
    # drain runs (the post-semaphore hazard the snapshot exists for).
    a.context._ctx_pressure_lockdown = False
    gen = out[0] if isinstance(out, tuple) else out
    async for _ in gen:
        pass

    assert calls, "derived-mood hook never fired on the full streamed turn"
    assert calls[0]["pressure_lockdown"] is True


def test_post_turn_hook_gated_off_for_sim_origin(tmp_path: Path):
    """§4AT-G class: a sim/self-play turn must never author production
    mood. The hook shares the capture site's real_only origin gate."""
    from ghost_agent.core.agent import GhostAgent
    agent = _make_traj_agent(tmp_path)
    agent.context.turn_origin_label = "sim"
    calls = []
    agent.context.self_model.update_derived_mood = (
        lambda **kw: calls.append(kw))
    GhostAgent._record_turn_trajectory(
        agent,
        messages=[{"role": "user", "content": "synthetic challenge"}],
        final_content="done", req_id="r3", model="m",
        user_request="synthetic challenge",
    )
    assert calls == []


# ---------------------------------------------------------------------------
# Wiring: biological watchdog phase 2.8c
# ---------------------------------------------------------------------------

def _make_bio_ctx(*, idle_secs: float, self_model, collector="default"):
    """Minimal mock context for _biological_tick — mirrors
    test_selfhood_biological_phase._make_ctx, except the trajectory
    collector defaults to a MagicMock: phase 2.8c is collector-gated
    (review R4 — no minting when the retiring half is unreachable), so
    the mood-phase tests need one present; pass collector=None to test
    the gate itself."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=idle_secs))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.prm_train_cooldown = None
    ctx.args.self_narrative_cooldown = None
    ctx.frontier_tracker = None
    ctx.reflector = None
    ctx.trajectory_collector = (MagicMock() if collector == "default"
                                else collector)
    ctx.prm_scorer = None
    # The MagicMock collector would otherwise awaken the postmortem
    # phase mock-driven (awaiting a MagicMock → caught TypeError +
    # warning noise on every tick test).
    ctx.postmortem_engine = None
    ctx.self_model = self_model
    return ctx


async def _bio_tick(ctx):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    return agent


async def test_idle_phase_refreshes_derived_mood():
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)
    agent = await _bio_tick(ctx)
    sm.update_derived_mood.assert_called_once()
    assert sm.update_derived_mood.call_args.kwargs["idle_seconds"] >= 1200
    assert agent._last_derived_mood_at > datetime.datetime.min


async def test_idle_phase_respects_cooldown():
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)
    agent = await _bio_tick(ctx)
    # Second tick on the SAME agent inside the cooldown → no second call.
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1300))
    await agent._biological_tick()
    assert sm.update_derived_mood.call_count == 1


async def test_idle_phase_skipped_below_window():
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=600, self_model=sm)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_not_called()


async def test_idle_phase_skipped_above_window():
    """The 15-60min band is reserved: >60min idle belongs to the deep
    phases (review R1: the upper bound was untested — dropping it would
    leak 2.8c into the >1h band)."""
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=4000, self_model=sm)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_not_called()


async def test_idle_phase_gated_off_for_sim_context():
    """Phase 2.8c shares the real_only origin rule: a harness driving
    _biological_tick with a sim-flavored context (read-only skill
    store) must not author production mood (review R1)."""
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)
    ctx.turn_origin_label = None
    ctx.skill_memory = SimpleNamespace(is_read_only=True)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_not_called()


async def test_idle_phase_skipped_when_selfhood_disabled():
    sm = MagicMock()
    sm.enabled = False
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_not_called()


async def test_idle_phase_gated_off_without_trajectory_collector():
    """Review R4: under --no-trajectories the post-turn hook (and with
    it, retirement) is unreachable — minting "idle" here would let a
    falsified label stand its full 48h TTL. No collector → no derived
    mood at all (symmetric death)."""
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm, collector=None)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_not_called()


async def test_idle_phase_resamples_idle_clock_mid_tick():
    """Review R4: idle_secs is measured at tick-top and phases 2.5-2.8
    run LLM calls before 2.8c — a user turn completing during those
    awaits must not be greeted by a stale >30min reading (the same
    re-sample fix phase 3b carries). Simulate the mid-tick turn by
    resetting last_activity_time from inside phase 2.8's await."""
    from unittest.mock import AsyncMock
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)

    async def _user_returns_mid_tick(**kw):
        ctx.last_activity_time = datetime.datetime.now()
        return ""

    sm.consolidate_narrative = AsyncMock(side_effect=_user_returns_mid_tick)
    await _bio_tick(ctx)
    sm.update_derived_mood.assert_called_once()
    assert sm.update_derived_mood.call_args.kwargs["idle_seconds"] < 100


async def test_backwards_clock_step_cannot_masquerade_as_post_turn(tmp_path: Path):
    """Executed pin for the 2.8c clamp floor (review R6: the 0.001→0.0
    mutant survived the whole suite). A backwards wall-clock step
    mid-tick makes the re-sampled idle negative; clamped to exactly 0.0
    it would hit the post-turn sentinel and let the lockdown-blind tick
    retire a standing "overloaded". The 0.001 floor keeps tick
    semantics: the mood must survive."""
    from unittest.mock import AsyncMock
    sm = SelfModel(root=tmp_path)
    m = sm.update_derived_mood(pressure_lockdown=True, operator_turn=True)
    assert m.label == "overloaded"
    ctx = _make_bio_ctx(idle_secs=1200, self_model=sm)

    async def _clock_steps_back(**kw):
        ctx.last_activity_time = (datetime.datetime.now()
                                  + datetime.timedelta(seconds=600))
        return ""

    sm.consolidate_narrative = AsyncMock(side_effect=_clock_steps_back)
    await _bio_tick(ctx)
    assert sm.state.mood() is not None
    assert sm.state.mood().label == "overloaded"


async def test_idle_phase_normalizes_bio_time_scale():
    """Review R4: the phase WINDOW is _bio_scaled but IDLE_AFTER_SECONDS
    is a production constant — without normalization "idle" is
    underivable under --bio-time-scale (the §4CB "scaled gates, not
    work" class). At scale 60, 30 real seconds of idle must reach the
    updater as ≥1800 effective seconds."""
    from ghost_agent.core.agent import GhostAgent
    sm = MagicMock()
    sm.enabled = True
    ctx = _make_bio_ctx(idle_secs=30, self_model=sm)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._bio_time_scale = 60.0
    await agent._biological_tick()
    sm.update_derived_mood.assert_called_once()
    assert sm.update_derived_mood.call_args.kwargs["idle_seconds"] >= 1700
