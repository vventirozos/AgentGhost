"""§4TC (2026-11-06): the mood is a pair of numbers that moves by itself.

The derived mood it replaces changed 148 times in three months and only ever
said idle / satisfied / stuck; it keyed on verifier verdicts that are mostly
off. The new one is `selfhood/mood.py` (the arithmetic — pure) and
`selfhood/state.py` (the stored pair, the journal).

Pinned here, on results only: decay and its half-life, the intensity
thresholds and their margin, the eight regions and theirs, the "curious" tag,
every event's sign, size and cap, boredom, the owner's anchors and synonyms,
the journal rows and their bound, what a read may and may not write, and old
files on disk.

All owner data here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood.schema import MOOD_RECENT_CAUSES, Mood, SelfState
from ghost_agent.selfhood.state import (
    DECAY_CAUSE, MOOD_HISTORY_FILENAME, OWNER_CAUSE, STATE_FILENAME, MoodView, SelfStateThread)

T0 = dt.datetime(2026, 11, 6, 9, 0, tzinfo=dt.timezone.utc)
UTC = dt.timezone.utc


def _at(hours: float = 0.0, minutes: float = 0.0) -> dt.datetime:
    return T0 + dt.timedelta(hours=hours, minutes=minutes)


def _rows(s: SelfStateThread) -> list:
    p = s.mood_history_path
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _put(s: SelfStateThread, p: float, e: float, now=T0, cause="seed", kind=M.KIND_TURN, **kw):
    """Move a neutral mood to (p, e) through the one writer."""
    return s.nudge(p, e, cause, kind, now=now, **kw)


# ── decay ───────────────────────────────────────────────────────────────────

def test_a_value_halves_every_half_life():
    assert M.HALF_LIFE_S == 7200.0
    assert M.decay(0.8, M.HALF_LIFE_S) == pytest.approx(0.4)
    assert M.decay(-0.8, 2 * M.HALF_LIFE_S) == pytest.approx(-0.2)
    assert M.decay(0.8, M.HALF_LIFE_S / 2) == pytest.approx(0.8 * 0.5 ** 0.5)


@pytest.mark.parametrize("span", [0, -5, None, "x", float("nan")])
def test_no_time_or_an_unreadable_span_decays_nothing(span):
    assert M.decay(0.8, span) == 0.8


def test_the_stored_pair_decays_when_it_is_read_and_nothing_is_written(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, 0.8, 0.0)
    raw = s.path.read_bytes()
    v = s.mood(now=_at(hours=2))
    assert (v.p, v.e) == (pytest.approx(0.4), 0.0)
    assert s.path.read_bytes() == raw and len(_rows(s)) == 1         # a read is a read
    assert s.state.mood.p == 0.8                                      # the stored pair is untouched


def test_decay_never_changes_the_label_only_the_intensity(tmp_path):
    """Both numbers halve alike, so the region is the same at every age."""
    s = SelfStateThread(tmp_path)
    _put(s, -0.9, 0.45)
    seen = [(s.mood(now=_at(hours=h)).label, s.mood(now=_at(hours=h)).intensity)
            for h in (0, 1, 2, 3, 4, 5, 6)]
    assert [lab for lab, n in seen if n] == ["irritable"] * sum(1 for _l, n in seen if n)
    assert [n for _l, n in seen] == [3, 2, 2, 1, 1, 0, 0]
    assert seen[-1][0] == "neutral"


def test_a_mood_whose_age_cannot_be_read_is_not_presented_as_current(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, -0.9, 0.45)
    s.state.mood.updated_at = "yesterday-ish"
    assert s.mood(now=T0).neutral


# ── intensity: thresholds and their margin ──────────────────────────────────

def test_the_thresholds():
    assert M.INTENSITY_THRESHOLDS == (0.20, 0.45, 0.75)
    assert [M.level_for(m) for m in (0.0, 0.19, 0.20, 0.44, 0.45, 0.74, 0.75, 1.0)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert M.level_for(-0.8) == 3                                     # a magnitude


def test_an_intensity_is_given_up_only_below_its_threshold_minus_the_margin():
    assert M.INTENSITY_MARGIN == 0.04
    assert M.level_for(0.72, held=3) == 3 and M.level_for(0.709, held=3) == 2
    assert M.level_for(0.42, held=2) == 2 and M.level_for(0.409, held=2) == 1
    assert M.level_for(0.17, held=1) == 1 and M.level_for(0.159, held=1) == 0
    # every level on the way down keeps its margin, so a mood that only decays
    # reads the same whenever it is looked at (3 → 2 is not skipped to 1)
    assert M.level_for(0.42, held=3) == 2 and M.level_for(0.17, held=3) == 1
    # … and far below, what was held does not matter
    assert M.level_for(0.30, held=3) == 1 and M.level_for(0.0, held=3) == 0
    # rising needs no margin, and a held level never lowers a higher reading
    assert M.level_for(0.46, held=1) == 2 and M.level_for(0.80, held=0) == 3


@pytest.mark.parametrize("held", [None, "x", -3, 9])
def test_a_held_level_that_is_no_level_is_read_safely(held):
    assert M.level_for(0.5, held=held) in (2, 3) and M.level_for(0.1, held=held) == 0


def test_the_automatic_mood_has_no_cap_and_reaches_three(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=6), now=T0)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=6), now=_at(minutes=1))
    assert (after.label, after.intensity, after.source) == ("irritable", 3, "auto")


# ── the regions ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,e,label", [
    (0.6, 0.45, "cheerful"), (0.6, 0.0, "content"), (0.6, -0.45, "calm"),
    (0.0, 0.6, "restless"), (0.0, -0.6, "bored"),
    (-0.6, 0.45, "irritable"), (-0.6, 0.0, "down"), (-0.45, -0.6, "tired"),
    # "near 0" is relative: the smaller under AXIS_RATIO of the larger
    (0.6, 0.23, "content"), (0.6, 0.25, "cheerful"), (0.6, -0.23, "content"), (0.6, -0.25, "calm"),
    (0.23, 0.6, "restless"), (0.25, 0.6, "cheerful"), (-0.23, -0.6, "bored"), (-0.25, -0.6, "tired"),
    (-0.6, 0.23, "down"), (-0.6, 0.25, "irritable"),
    (0.5, 0.5, "cheerful"), (-0.5, -0.5, "tired"), (0.5, -0.5, "calm"), (-0.5, 0.5, "irritable"),
    (0.0, 0.0, "neutral"),
])
def test_the_label_is_read_off_the_signs_and_the_relative_size(p, e, label):
    assert M.AXIS_RATIO == 0.40
    assert M.region(p, e) == label


def test_every_region_is_one_of_the_eight_labels_and_all_eight_are_reachable():
    seen = {M.region(p / 10, e / 10) for p in range(-10, 11) for e in range(-10, 11)}
    assert seen == set(M.LABELS[:8]) | {"neutral"}
    assert M.LABELS == ("cheerful", "content", "calm", "restless", "bored",
                        "irritable", "down", "tired", "curious")


def test_a_held_label_stands_until_the_pair_is_clearly_past_the_boundary():
    assert M.AXIS_MARGIN == 0.10
    # content ↔ cheerful: the boundary is ratio 0.40, held to 0.50 / 0.30
    assert M.held_region(0.6, 0.27, "content") == "content"         # ratio 0.45: raw says cheerful
    assert M.held_region(0.6, 0.31, "content") == "cheerful"        # ratio 0.52: clearly past
    assert M.held_region(0.6, 0.21, "cheerful") == "cheerful"       # ratio 0.35: raw says content
    assert M.held_region(0.6, 0.17, "cheerful") == "content"        # ratio 0.28
    # a sign flip is always past the boundary
    assert M.held_region(0.6, -0.3, "cheerful") == "calm"
    assert M.held_region(-0.6, 0.0, "content") == "down"
    # nothing held, or a word that is no region: the plain reading
    assert M.held_region(0.6, 0.27, "") == "cheerful" and M.held_region(0.6, 0.27, "sparkly") == "cheerful"
    # the margin is one-sided: a held label never claims a pair deep in another region
    assert M.held_region(0.2, 0.6, "content") == "restless"


def test_a_small_wobble_across_a_boundary_does_not_flip_the_stored_label(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, 0.60, 0.20)                                               # content (ratio 0.33)
    _b, after = s.nudge(0.0, 0.06, "wobble", M.KIND_TURN, now=T0)     # ratio 0.43: just past
    assert after.label == "content" and len(_rows(s)) == 1
    _b, after = s.nudge(0.0, 0.08, "push", M.KIND_TURN, now=T0)       # ratio 0.57: clearly past
    assert after.label == "cheerful" and [r["label"] for r in _rows(s)] == ["content", "cheerful"]


# ── "curious": pleasant and energetic, reached through novelty ──────────────

def test_curious_is_a_pleasant_energetic_mood_reached_through_a_novelty_event(tmp_path):
    s = SelfStateThread(tmp_path)
    _b, after = s.nudge(0.3, 0.3, "read 5 sources in a research turn", M.KIND_TURN, novelty=True, now=T0)
    assert after.label == "curious"
    plain = SelfStateThread(tmp_path / "plain")
    assert plain.nudge(0.3, 0.3, "x", M.KIND_TURN, now=T0)[1].label == "cheerful"


def test_decay_keeps_the_tag_and_trouble_drops_it(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.6, 0.6, "research", M.KIND_TURN, novelty=True, now=T0)
    assert s.mood(now=_at(hours=1)).label == "curious"                # decay keeps the tag
    stepped = s.settle(now=_at(hours=2))                              # … also once a step down is written
    assert (stepped.label, stepped.intensity) == ("curious", 1) and s.state.mood.novelty is True
    assert s.nudge(-0.9, 0.3, "trouble", M.KIND_TURN, now=_at(hours=2, minutes=1))[1].label == "irritable"
    back = s.nudge(1.5, 0.0, "x", M.KIND_TURN, now=_at(hours=2, minutes=2))[1]
    assert back.label == "cheerful"                                   # the tag does not come back by itself


def test_the_tag_belongs_to_the_move_that_set_it(tmp_path):
    """Review r2: research, then three thumbs, read "curious, 3 of 3: the
    owner gave a thumbs-up". Any later event that is not itself a research
    event clears the tag."""
    s = SelfStateThread(tmp_path)
    assert s.nudge(0.15, 0.20, "read 4 sources in a research turn", M.KIND_TURN, novelty=True, now=T0)[1].label == "curious"
    labels = [s.nudge(0.25, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP, now=_at(minutes=i))[1].label
              for i in (1, 2, 3)]
    assert "curious" not in labels and s.state.mood.novelty is False
    assert s.mood(now=_at(minutes=4)).label in ("content", "cheerful") and s.mood(now=_at(minutes=4)).intensity == 3
    again = s.nudge(0.15, 0.20, "read 3 sources in a research turn", M.KIND_TURN, novelty=True, now=_at(minutes=5))[1]
    assert again.label == "curious"                                   # a new research event tags again
    for kind, move in ((M.KIND_PRAISE, (0.10, 0.0)), (M.KIND_TURN, (0.08, 0.0))):
        t = SelfStateThread(tmp_path / kind)
        t.nudge(0.3, 0.3, "research", M.KIND_TURN, novelty=True, now=T0)
        assert t.nudge(*move, "later", kind, now=T0)[1].label == "cheerful", kind


def test_the_owners_word_clears_an_old_tag_and_a_research_turn_after_it_may_tag(tmp_path):
    """(The review's mutant R21 was alive: nothing pinned that `set_owner_mood`
    clears the tag.)"""
    s = SelfStateThread(tmp_path)
    s.nudge(0.3, 0.3, "research", M.KIND_TURN, novelty=True, now=T0)
    _b, after = s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, now=T0)
    assert after.label == "cheerful" and s.state.mood.novelty is False
    assert SelfStateThread(tmp_path).mood(now=T0).label == "cheerful"                 # read off the stored pair
    assert json.loads(s.path.read_text())["mood"]["novelty"] is False
    _b, tagged = s.nudge(0.15, 0.20, "read 4 sources in a research turn", M.KIND_TURN, novelty=True, now=T0)
    assert tagged.label == "curious"                                  # an owner-set mood, then research
    _b, his = s.set_owner_mood("curious", M.ANCHOR_DIRECTION["curious"], 2, now=T0)
    assert his.label == "curious" and s.state.mood.novelty is True    # unless curious is what he asked for


def test_a_novelty_event_that_leaves_the_mood_unpleasant_tags_nothing(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, -0.9, 0.0)
    assert s.nudge(0.1, 0.5, "research", M.KIND_TURN, novelty=True, now=T0)[1].label == "irritable"
    assert s.nudge(1.6, 0.0, "x", M.KIND_TURN, now=T0)[1].label == "cheerful"      # P+ E+ now, untagged
    assert s.state.mood.novelty is False


@pytest.mark.parametrize("p,e,label", [
    (0.03, 0.28, "curious"),          # pleasant by a little: the tag holds (untagged it reads "restless")
    (0.28, 0.03, "curious"),          # … and so on the other side (untagged: "content")
    (0.30, 0.30, "curious"),
    (-0.03, 0.28, "restless"),        # trouble won the sign of P: no tag
    (0.0, 0.28, "restless"),
    (0.28, -0.03, "content"),         # the energy is gone: no tag
    (0.28, 0.0, "content"),
    (-0.3, -0.3, "tired"),
])
def test_the_tag_needs_both_numbers_positive(p, e, label):
    assert M.read_off(p, e, novelty=True)[0] == label
    assert M.read_off(p, e, novelty=True)[2] is (label == "curious")


def test_a_research_turn_with_one_failure_is_still_curious_and_with_two_it_is_not(tmp_path):
    """Review: research plus ONE failed call read "restless" — the tag was
    lost because the pair left the diagonal. The precedence: trouble wins the
    sign of P; the tag stays only while P ends positive."""
    one = SelfModel(root=tmp_path / "one")
    after = one.note_owner_turn(M.TurnSignals(tool_calls=5, failures=1, sources=4), now=T0)
    assert (after.p, after.e) == (pytest.approx(0.03), pytest.approx(0.28))
    assert (after.label, after.intensity) == ("curious", 1)
    two = SelfModel(root=tmp_path / "two")
    after = two.note_owner_turn(M.TurnSignals(tool_calls=5, failures=2, sources=4), now=T0)
    assert after.p == pytest.approx(-0.09) and (after.label, after.intensity) == ("restless", 1)
    assert two.state.state.mood.novelty is False


# ── events: what one owner turn does ────────────────────────────────────────

def _one(sig) -> M.Event:
    evs = M.turn_events(sig)
    assert len(evs) == 1, evs
    return evs[0]


def test_a_plain_chat_turn_changes_nothing(tmp_path):
    assert M.turn_events(M.TurnSignals()) == []
    assert M.turn_events(M.TurnSignals(tool_calls=2, duration_s=40.0)) == []
    sm = SelfModel(root=tmp_path)
    assert sm.note_owner_turn(M.TurnSignals(), now=T0) is None
    assert not sm.state.path.exists() and not sm.state.mood_history_path.exists()


def test_each_failed_call_lowers_pleasantness_and_raises_energy():
    ev = _one(M.TurnSignals(tool_calls=5, failures=1))
    assert (ev.dp, ev.de) == M.EV_TROUBLE == (-0.12, 0.08)
    ev = _one(M.TurnSignals(tool_calls=5, failures=3))
    assert (ev.dp, ev.de) == (pytest.approx(-0.36), pytest.approx(0.24))
    assert ev.cause == "3 tool calls failed in this turn" and ev.kind == M.KIND_TURN and not ev.novelty


def test_a_guard_event_counts_like_a_failure_and_is_named():
    ev = _one(M.TurnSignals(tool_calls=5, guards=1))
    assert (ev.dp, ev.de) == M.EV_TROUBLE and ev.cause == "a loop guard stepped in once in this turn"
    ev = _one(M.TurnSignals(tool_calls=5, failures=1, guards=2))
    assert (ev.dp, ev.de) == (pytest.approx(-0.36), pytest.approx(0.24))
    assert ev.cause == "1 tool call failed and a loop guard stepped in 2 times in this turn"


def test_a_turn_that_ended_under_context_pressure_is_named_as_that():
    """Review: it was journalled as "a loop guard stepped in"."""
    ev = _one(M.TurnSignals(tool_calls=2, pressure=True))
    assert (ev.dp, ev.de) == M.EV_TROUBLE
    assert ev.cause == "the context ran out of room (context pressure) in this turn" and "guard" not in ev.cause
    ev = _one(M.TurnSignals(tool_calls=5, failures=2, guards=1, pressure=True))
    assert (ev.dp, ev.de) == (pytest.approx(-0.48), pytest.approx(0.32))
    assert ev.cause == ("2 tool calls failed and a loop guard stepped in once and the context ran out of room "
                        "(context pressure) in this turn")
    assert M.turn_events(M.TurnSignals(tool_calls=2, pressure=False)) == []


def test_a_turn_in_which_the_owner_set_the_mood_gives_no_event():
    """His word holds for the turn."""
    busy = dict(tool_calls=30, failures=6, guards=2, sources=9, duration_s=4000.0, pressure=True)
    assert len(M.turn_events(M.TurnSignals(**busy))) == 3
    assert M.turn_events(M.TurnSignals(owner_set=True, **busy)) == []
    assert M.merge_events(M.turn_events(M.TurnSignals(owner_set=True, **busy))) is None


def test_trouble_is_capped_per_turn():
    assert M.TROUBLE_CAP == 4
    capped = _one(M.TurnSignals(tool_calls=20, failures=9, guards=9, duration_s=10.0))
    assert (capped.dp, capped.de) == (pytest.approx(-0.48), pytest.approx(0.32))
    assert capped.cause.startswith("9 tool calls failed and a loop guard stepped in 9 times")
    at_cap = M.turn_events(M.TurnSignals(tool_calls=4, failures=4))[0]
    assert (at_cap.dp, at_cap.de) == (pytest.approx(capped.dp), pytest.approx(capped.de))


def test_a_clean_multi_tool_turn_lifts_pleasantness_a_little():
    ev = _one(M.TurnSignals(tool_calls=3, duration_s=20.0))
    assert (ev.dp, ev.de) == M.EV_CLEAN_TURN == (0.08, 0.0)
    assert ev.cause == "a turn of 3 tool calls went through without a failure"
    assert M.CLEAN_TURN_MIN_CALLS == 3
    assert M.turn_events(M.TurnSignals(tool_calls=2)) == []                         # too few calls
    assert [e.dp for e in M.turn_events(M.TurnSignals(tool_calls=5, failures=1))] == [-0.12]      # one failed
    assert [e.dp for e in M.turn_events(M.TurnSignals(tool_calls=5, guards=1))] == [-0.12]        # a guard
    assert M.turn_events(M.TurnSignals(tool_calls=5, completed=False)) == []        # it was cut off


def test_a_research_turn_lifts_both_and_carries_the_novelty_tag():
    assert M.RESEARCH_MIN_SOURCES == 3
    evs = M.turn_events(M.TurnSignals(tool_calls=2, sources=3))
    assert [(e.dp, e.de, e.novelty) for e in evs] == [(0.15, 0.20, True)] and M.EV_RESEARCH == (0.15, 0.20)
    assert evs[0].cause == "read 3 sources in a research turn"
    assert M.turn_events(M.TurnSignals(tool_calls=2, sources=2)) == []
    # it does not need a clean turn
    kinds = [(e.dp, e.novelty) for e in M.turn_events(M.TurnSignals(tool_calls=6, failures=1, sources=5))]
    assert kinds == [(-0.12, False), (0.15, True)]


def test_a_very_long_turn_lowers_energy_and_is_tired_not_bored(tmp_path):
    assert (M.LONG_TURN_S, M.LONG_TURN_CALLS) == (900.0, 25)
    ev = _one(M.TurnSignals(tool_calls=2, duration_s=900.0))
    assert (ev.dp, ev.de) == M.EV_LONG_TURN == (-0.15, -0.30)
    assert ev.cause == "a very long turn (15 min, 2 tool calls)"
    by_calls = _one(M.TurnSignals(tool_calls=25, duration_s=60.0))
    assert (by_calls.dp, by_calls.de) == M.EV_LONG_TURN
    assert M.turn_events(M.TurnSignals(tool_calls=2, duration_s=899.0)) == []
    assert [e.dp for e in M.turn_events(M.TurnSignals(tool_calls=24, duration_s=60.0))] == [0.08]
    # a long turn is not also a "clean turn": alone it reads tired
    sm = SelfModel(root=tmp_path)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=30, duration_s=1200.0), now=T0)
    assert (after.label, after.intensity) == ("tired", 1)


def test_one_turns_events_are_one_move_with_the_causes_joined(tmp_path):
    sm = SelfModel(root=tmp_path)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=2, sources=4), now=T0)
    assert (after.p, after.e) == (pytest.approx(-0.24 + 0.15), pytest.approx(0.16 + 0.20))
    rows = _rows(sm.state)
    assert len(rows) == 1 and rows[0]["kind"] == "turn"
    assert rows[0]["cause"] == "2 tool calls failed in this turn; read 4 sources in a research turn"
    assert M.merge_events([]) is None and M.merge_events([None]) is None


@pytest.mark.parametrize("sig", [
    M.TurnSignals(tool_calls=-3, failures=-1, guards=-2, sources=-9, duration_s=-5.0),
    M.TurnSignals(tool_calls=None, failures=None, guards=None, sources=None, duration_s=None),
    M.TurnSignals(duration_s="soon"),
])
def test_unreadable_signals_are_no_events(sig):
    assert M.turn_events(sig) == []


# ── events: the owner's reactions ───────────────────────────────────────────

@pytest.mark.parametrize("kind,move,cause", [
    (M.KIND_THUMB_UP, (0.25, 0.0), "the owner gave a thumbs-up"),
    (M.KIND_THUMB_DOWN, (-0.30, 0.0), "the owner gave a thumbs-down"),
    (M.KIND_CORRECTION, (-0.20, 0.0), "the owner corrected my last reply"),
    (M.KIND_PRAISE, (0.10, 0.0), "the owner said thanks"),
])
def test_each_reaction_has_its_sign_size_and_cause(tmp_path, kind, move, cause):
    ev = M.reaction_event(kind)
    assert (ev.dp, ev.de, ev.cause, ev.kind, ev.novelty) == (*move, cause, kind, False)
    sm = SelfModel(root=tmp_path)
    after = sm.note_owner_reaction(kind, now=T0)
    assert (after.p, after.e) == move and after.recent[-1] == {"at": "2026-11-06T09:00:00Z", "cause": cause, "kind": kind}


def test_the_owners_reactions_move_the_pleasantness_only():
    """Review r2: the thumbs-up carried +0.05 of energy, so toggling up / down
    on one turn pumped the energy."""
    for kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN, M.KIND_CORRECTION, M.KIND_PRAISE):
        assert M.reaction_event(kind).de == 0.0, kind


def test_toggling_a_thumb_ten_times_leaves_the_energy_where_it_was(tmp_path):
    sm = SelfModel(root=tmp_path)
    for i in range(10):
        sm.note_owner_reaction(M.KIND_THUMB_UP, now=_at(minutes=i))
        sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=_at(minutes=i, hours=0))
    assert sm.mood(now=_at(minutes=10)).e == 0.0


def test_a_thumb_is_stronger_than_thanks_and_a_thumbs_down_than_a_correction():
    assert M.EV_THUMB_UP[0] > M.EV_PRAISE[0] > 0 > M.EV_CORRECTION[0] > M.EV_THUMB_DOWN[0]
    assert M.level_for(M.EV_PRAISE[0]) == 0 and M.level_for(2 * M.EV_PRAISE[0]) == 1      # "a little"


@pytest.mark.parametrize("kind", ["", "thumb", "owner", "quiet", None, 7])
def test_an_unknown_reaction_moves_nothing(tmp_path, kind):
    assert M.reaction_event(kind) is None
    sm = SelfModel(root=tmp_path)
    assert sm.note_owner_reaction(kind, now=T0) is None and not sm.state.path.exists()


@pytest.mark.parametrize("text", [
    "thanks", "Thanks!", "thank you", "THANK YOU", "thx", "great", "Perfect.", "awesome", "nice", "bravo!",
    "well done", "nice   work", "good job", "great job", "ευχαριστώ", "Ευχαριστώ πολύ", "μπράβο", "μπραβο",
    "τέλεια", "ok thanks", "ok, thanks!", "thanks ghost", "that's great", "thank you very much",
    "thanks a lot", "thank you so much \U0001f64f", "really nice", "perfect, thanks", "10/10 perfect",
    "that\u2019s great",                                          # a curly apostrophe is a plain one
    # review r2: common forms that were missing
    "σε ευχαριστώ", "σ'ευχαριστώ", "Σ\u2019ευχαριστώ πολύ", "μπράβο σου", "ευχαριστώ ρε", "great work",
    "good work", "many thanks", "thanks mate", "thank u", "cheers", "Cheers!", "many thanks, ghost!",
    "\"thanks!\"", "great!!", "thanks :)",
])
def test_a_short_message_of_thanks_and_filler_only_is_praise(text):
    assert M.has_praise(text) is True


@pytest.mark.parametrize("text", [
    # the review's list: six of eighteen live matches of a plain word list were not praise
    "make it look awesome", "\u2026this wasn't awesome", "a great number of files",
    "thanks to the cache it works", "Great Britain's capital?", "perfect square numbers",
    "oh great, it broke again", "no, thanks", "that isn\u2019t great", "δεν είναι τέλεια", "όχι ευχαριστώ",
    "Here is the report on Bruno's vet visit, as promised. " * 6 + "thank you",
    # … and the edges of the rule
    "", "   ", None, 7, ["thanks"], "ok", "ghost", "that's very much", "thank", "you", "good", "job done",
    "thank you, Bruno loved it", "ok, thanks \u2014 now the vet's number please", "thanksgiving plans",
    "the greatest hits", "imperfect", "not great", "that was not perfect", "never great", "wellness done",
    "what is Bruno's age?", "great?? why did it fail", "thanks thanks thanks thanks thanks thanks thanks",
    # review r2: a question is not praise, and "is" / "it's" / "that" are not filler
    "great?", "thanks?", "is that perfect", "τέλεια;", "τέλεια\u037e", "is that so great", "that is great",
    "it's perfect", "that great", "great? \"", "perfect?!", "thanks? \U0001f642", "well done?)", "ευχαριστώ;",
    "is", "that", "mate", "ρε", "σου", "many", "u",
])
def test_everything_else_is_not_praise(text):
    assert M.has_praise(text) is False


def test_a_negation_is_never_filler():
    """What keeps "no, thanks" and "not great" out is that a negation is
    neither praise nor filler — so none may ever be added to those lists."""
    negations = {"no", "not", "never", "nope", "isn't", "wasn't", "don't", "δεν", "δε", "όχι", "οχι", "μη", "μην"}
    assert not negations & (M.PRAISE_WORDS | M.PRAISE_FILLER)
    assert not negations & {w for pair in M.PRAISE_PAIRS for w in pair}
    for neg in sorted(negations):
        assert M.has_praise(f"{neg} great") is False and M.has_praise(f"great {neg}") is False, neg


def test_praise_is_at_most_six_words():
    assert M.PRAISE_MAX_WORDS == 6
    assert M.has_praise("ok that's really very nice ghost") is True              # six
    assert M.has_praise("ok ok that's really very nice ghost") is False          # seven


# ── quiet → boredom: read-time only ─────────────────────────────────────────
#
# The first rule (energy pulled down after 3 quiet hours, written into the pair
# when the owner returned) left 13 % of the owner's turns starting neutral and
# about 40 % starting "bored" when replayed over a month of his real turns.
# The rule now: boredom is what a reader OTHER than the owner sees over a
# neutral mood after 6 quiet hours (1) and 12 (2). It is never in the pair.

H = 3600.0


def test_boredom_is_one_after_six_quiet_hours_two_after_twelve_and_never_more():
    assert (M.BOREDOM_AFTER_S, M.BOREDOM_DEEP_AFTER_S) == (6 * H, 12 * H)
    assert [M.boredom_level(q) for q in (0, 3 * H, 6 * H - 1, 6 * H, 9 * H, 12 * H - 1, 12 * H, 30 * 24 * H)] == [
        0, 0, 0, 1, 1, 1, 2, 2]
    assert M.quiet_cause(1) == "quiet for 6 hours" and M.quiet_cause(2) == "quiet for 12 hours"


@pytest.mark.parametrize("quiet", [None, "x", 0, -1, 60.0, float("nan"), [7], 5.9 * 3600])
def test_an_unknown_or_short_quiet_is_no_boredom(quiet):
    assert M.boredom_level(quiet) == 0


def test_a_reader_sees_boredom_over_a_neutral_mood_and_the_pair_is_never_touched(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)                                          # the owner was last here at T0
    raw = s.path.read_bytes()
    seen = [(s.mood(now=_at(hours=h)).label, s.mood(now=_at(hours=h)).intensity) for h in (1, 5.9, 6, 11.9, 12, 400)]
    assert seen == [("neutral", 0), ("neutral", 0), ("bored", 1), ("bored", 1), ("bored", 2), ("bored", 2)]
    v = s.mood(now=_at(hours=13))
    assert (v.p, v.e, v.source, v.cause) == (0.0, 0.0, "auto", "quiet for 12 hours")
    assert s.path.read_bytes() == raw and s.state.mood is None        # nothing of it is stored


def test_a_real_mood_that_has_not_worn_off_is_not_replaced_by_boredom(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    _put(s, -0.9, 0.45, now=_at(hours=5))                              # e.g. a thumb that came without a turn
    assert s.mood(now=_at(hours=7)).label == "irritable"               # quiet for 7 h, and still irritable
    assert s.mood(now=_at(hours=9)).label == "irritable"
    assert s.mood(now=_at(hours=12)).label == "bored"                  # worn off — and the quiet is long
    owner = SelfStateThread(tmp_path / "owner")                        # the same for a mood the owner set
    owner.mark_owner_turn(now=T0)
    owner.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 3, now=_at(hours=5))
    seen = [(owner.mood(now=_at(hours=h)).label, owner.mood(now=_at(hours=h)).source) for h in (7, 9, 12)]
    assert seen == [("cheerful", "owner"), ("cheerful", "owner"), ("bored", "auto")]


def test_the_owners_own_request_is_never_answered_bored(tmp_path):
    """His message ended the quiet. Everyone else answered during it hears it."""
    from ghost_agent.utils.logging import request_id_context, requester_role_context
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    late = _at(hours=13)
    for rid, bored in (("web-elena-1", False), ("slack-elena-2", False), ("probe-1", True), ("sched-news", True),
                       ("job-4", True), ("SYSTEM", True)):
        tok = request_id_context.set(rid)
        try:
            assert (s.mood(now=late).label == "bored") is bored, rid
        finally:
            request_id_context.reset(tok)
    tok, role = request_id_context.set("slack-m-3"), requester_role_context.set("member")
    try:
        assert s.mood(now=late).label == "bored"                       # a member is not the owner
    finally:
        request_id_context.reset(tok)
        requester_role_context.reset(role)


def test_when_the_owner_returns_the_quiet_simply_ends(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    assert s.mood(now=_at(hours=13)).label == "bored"
    s.owner_arrived(now=_at(hours=13, minutes=5))
    after = s.mood(now=_at(hours=13, minutes=6))
    assert after.neutral and (after.p, after.e) == (0.0, 0.0)          # nothing was carried into his turn
    assert s.state.mood is None or (s.state.mood.p, s.state.mood.e) == (0.0, 0.0)
    assert s.mood(now=_at(hours=19, minutes=4)).neutral                # the next quiet starts from his return
    assert s.mood(now=_at(hours=19, minutes=6)).label == "bored"


def test_the_quiet_is_measured_from_a_stamp_that_survives_a_restart(tmp_path):
    s = SelfStateThread(tmp_path)
    assert s.quiet_seconds(now=T0) is None                             # never stamped, no clock: unknown
    s.mark_owner_turn(now=T0)
    assert json.loads(s.path.read_text())["last_owner_turn_at"] == "2026-11-06T09:00:00Z"
    again = SelfStateThread(tmp_path)                                  # a restart in the night
    assert again.quiet_seconds(now=_at(hours=7)) == 7 * H
    assert again.mood(now=_at(hours=7)).label == "bored"
    again.attach_owner_clock(lambda: 60.0)                             # the process's own clock says "just now"
    assert again.quiet_seconds(now=_at(hours=7)) == 7 * H              # … the stamp wins


def test_the_process_clock_is_the_fallback_while_there_is_no_stamp(tmp_path):
    s = SelfStateThread(tmp_path)
    s.attach_owner_clock(lambda: 7 * H)
    assert s.quiet_seconds(now=T0) == 7 * H and s.mood(now=T0).label == "bored"
    assert s.mood(now=T0, quiet_s=0.0).neutral                         # an explicit quiet wins

    def broken():
        raise RuntimeError("clock gone")
    s.attach_owner_clock(broken)
    assert s.quiet_seconds(now=T0) is None and s.mood(now=T0).neutral  # unknown quiet → no boredom
    s.attach_owner_clock("not callable")
    assert s.mood(now=T0).neutral
    s.attach_owner_clock(lambda: None)
    assert s.mood(now=T0).neutral


@pytest.mark.parametrize("stamp", ["", "yesterday-ish", None, 1755000000])
def test_an_unreadable_stamp_is_no_stamp(tmp_path, stamp):
    (tmp_path / STATE_FILENAME).write_text(json.dumps({"last_owner_turn_at": stamp}))
    s = SelfStateThread(tmp_path)
    assert s.quiet_seconds(now=T0) is None and s.mood(now=T0).neutral


# ── the journal of a quiet: one row per level, one when it ends ─────────────

def test_the_first_read_to_see_each_level_writes_its_row_dated_when_the_threshold_was_crossed(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=5))
    assert _rows(s) == []
    for h in (7.5, 8, 9.25, 11.9):                                     # /api/health polling through level 1
        assert s.mood(now=_at(hours=h)).set_at == "2026-11-06T15:00:00Z"
    assert [(r["label"], r["intensity"], r["kind"], r["cause"], r["at"]) for r in _rows(s)] == [
        ("bored", 1, "quiet", "quiet for 6 hours", "2026-11-06T15:00:00Z")]
    for h in (12.5, 13, 40):
        assert s.mood(now=_at(hours=h)).set_at == "2026-11-06T21:00:00Z"
    assert [(r["intensity"], r["cause"], r["at"]) for r in _rows(s)] == [
        (1, "quiet for 6 hours", "2026-11-06T15:00:00Z"), (2, "quiet for 12 hours", "2026-11-06T21:00:00Z")]


def test_a_quiet_first_looked_at_after_twelve_hours_gets_both_rows(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=20))
    assert [(r["intensity"], r["at"]) for r in _rows(s)] == [(1, "2026-11-06T15:00:00Z"), (2, "2026-11-06T21:00:00Z")]


def test_a_restart_in_the_quiet_adds_no_second_row(tmp_path):
    """What the quiet already has is read off the last journal row."""
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=7))
    again = SelfStateThread(tmp_path)
    for h in (7.5, 8, 11):
        again.mood(now=_at(hours=h))
    assert len(_rows(again)) == 1
    again.mood(now=_at(hours=12.5))
    third = SelfStateThread(tmp_path)
    third.mood(now=_at(hours=13))
    assert [r["intensity"] for r in _rows(third)] == [1, 2]


def test_the_owners_return_writes_the_quiet_ended_once_and_only_after_a_bored_row(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.owner_arrived(now=_at(hours=4))                                  # nobody saw a boredom: no row
    assert _rows(s) == []
    s.mood(now=_at(hours=11))                                          # 7 h after his return
    s.owner_arrived(now=_at(hours=11, minutes=30))
    s.owner_arrived(now=_at(hours=11, minutes=31))                     # his next message: no second row
    assert [(r["label"], r["intensity"], r["kind"], r["cause"], r["at"]) for r in _rows(s)] == [
        ("bored", 1, "quiet", "quiet for 6 hours", "2026-11-06T19:00:00Z"),
        ("neutral", 0, "quiet_end", "the quiet ended", "2026-11-06T20:30:00Z")]
    assert M.QUIET_ENDED_CAUSE == "the quiet ended"


def _in_order(rows) -> bool:
    stamps = [M.parse_iso_utc(r["at"]) for r in rows]
    return all(a is not None for a in stamps) and all(a <= b for a, b in zip(stamps, stamps[1:]))


THUMB = (0.25, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP)


def test_a_thumb_in_the_middle_of_a_quiet_does_not_reopen_or_hide_it(tmp_path):
    """A reaction that comes without a turn (a thumb on yesterday's reply)
    lifts the mood for a while; the quiet goes on. ONE process: no second
    "bored 1" row, the thumb's mood gets its "it wore off" row when it ends —
    not at the owner's return — the rows are in time order, and the owner's
    return closes the quiet once."""
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=7))                                           # bored 1 (the memo knows this quiet)
    s.nudge(*THUMB, now=_at(hours=7, minutes=30))
    assert s.mood(now=_at(hours=8)).label == "content"                 # a real mood: not replaced
    back = s.mood(now=_at(hours=10))                                   # worn off, the quiet is still long
    assert (back.label, back.intensity) == ("bored", 1)
    rows = _rows(s)
    assert [(r["kind"], r["label"], r["intensity"]) for r in rows] == [
        ("quiet", "bored", 1), ("thumb_up", "content", 1), ("decay", "neutral", 0)]
    assert rows[2]["at"].startswith("2026-11-06T17:47:")               # when it wore off, not when it was seen
    assert back.set_at == rows[2]["at"]                                # bored again SINCE the thumb's mood ended
    s.mood(now=_at(hours=13))
    s.owner_arrived(now=_at(hours=14))
    rows = _rows(s)
    assert [r["kind"] for r in rows] == ["quiet", "thumb_up", "decay", "quiet", "quiet_end"]
    assert _in_order(rows), [r["at"] for r in rows]


def test_a_restart_after_a_thumb_still_finds_the_open_quiet(tmp_path):
    """The same, with the process restarted while the thumb's mood was alive:
    what the quiet has is read off the journal, past the thumb's rows."""
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=7))
    s.nudge(*THUMB, now=_at(hours=7, minutes=30))
    again = SelfStateThread(tmp_path)
    assert again.mood(now=_at(hours=10, minutes=30)).label == "bored"
    assert [r["kind"] for r in _rows(again)] == ["quiet", "thumb_up", "decay"]          # no second "bored 1"
    again.mood(now=_at(hours=13))
    again.owner_arrived(now=_at(hours=14))
    rows = _rows(again)
    assert [r["kind"] for r in rows] == ["quiet", "thumb_up", "decay", "quiet", "quiet_end"] and _in_order(rows)


def test_a_bored_row_is_never_dated_before_the_mood_it_followed_wore_off(tmp_path):
    """Review r2 (a): a thumb at +5 h whose mood wears off AFTER the six-hour
    line gave `13:00 content 1 → 14:17 neutral → 14:00 bored 1`. The bored row
    is dated at the later of the two moments, and the reading says the same."""
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)                                          # 09:00
    s.nudge(*THUMB, now=_at(hours=5))                                  # 14:00 content 1
    assert s.mood(now=_at(hours=6, minutes=5)).label == "content"      # past the line, the mood still alive
    seen = s.mood(now=_at(hours=8))
    rows = _rows(s)
    assert [(r["kind"], r["label"]) for r in rows] == [("thumb_up", "content"), ("decay", "neutral"),
                                                       ("quiet", "bored")]
    wore_off, bored = M.parse_iso_utc(rows[1]["at"]), M.parse_iso_utc(rows[2]["at"])
    assert rows[1]["at"].startswith("2026-11-06T15:17:")               # 0.25 → under 0.16: 1 h 17 min
    assert bored == wore_off > M.parse_iso_utc("2026-11-06T15:00:00Z") and _in_order(rows)
    assert seen.set_at == rows[2]["at"] and (seen.label, seen.intensity) == ("bored", 1)
    assert s.mood(now=_at(hours=9)).set_at == rows[2]["at"]            # the same on every later read
    s.mood(now=_at(hours=12, minutes=30))                              # bored 2: its own line, in order
    rows = _rows(s)
    assert rows[-1]["at"] == "2026-11-06T21:00:00Z" and _in_order(rows)


def test_two_levels_first_seen_after_a_late_thumb_are_both_dated_when_its_mood_ended(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.nudge(*THUMB, now=_at(hours=11, minutes=30))                     # wears off at +12 h 47 min
    s.mood(now=_at(hours=14))
    rows = _rows(s)
    assert [(r["kind"], r["intensity"]) for r in rows] == [("thumb_up", 1), ("decay", 0), ("quiet", 1), ("quiet", 2)]
    assert rows[2]["at"] == rows[3]["at"] == rows[1]["at"] and _in_order(rows)


def test_a_pair_that_never_became_a_mood_does_not_move_the_date_of_a_boredom(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.nudge(0.15, 0.0, "the owner said thanks", M.KIND_PRAISE, now=_at(hours=6, minutes=30))   # never a mood
    assert s.mood(now=_at(hours=7)).set_at == "2026-11-06T15:00:00Z"   # the threshold: nothing wore off
    assert _rows(s)[-1]["at"] == "2026-11-06T15:00:00Z"


def test_the_next_quiet_gets_rows_of_its_own(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    s.mood(now=_at(hours=7))
    s.owner_arrived(now=_at(hours=8))
    s.mark_owner_turn(now=_at(hours=8, minutes=10))                    # his turn ended
    s.mood(now=_at(hours=15))                                          # 6 h 50 min later
    assert [(r["kind"], r["at"]) for r in _rows(s)] == [
        ("quiet", "2026-11-06T15:00:00Z"), ("quiet_end", "2026-11-06T17:00:00Z"),
        ("quiet", "2026-11-06T23:10:00Z")]


def test_a_mood_that_wore_off_unseen_is_written_down_before_the_boredom(tmp_path):
    """The journal stays in order: "it wore off" (dated when it did), then
    "bored" (dated when the quiet reached six hours)."""
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    _put(s, -0.9, 0.45, cause="6 tool calls failed in this turn")
    s.mood(now=_at(hours=9))
    rows = _rows(s)
    assert [(r["label"], r["intensity"], r["kind"]) for r in rows] == [
        ("irritable", 3, "turn"), ("neutral", 0, "decay"), ("bored", 1, "quiet")]
    assert rows[1]["at"].startswith("2026-11-06T13:59:") and rows[2]["at"] == "2026-11-06T15:00:00Z"
    stored = json.loads(s.path.read_text())["mood"]
    assert (stored["label"], stored["intensity"]) == ("neutral", 0) and abs(stored["e"]) < 0.05
    assert stored["e"] > 0                                             # the decayed pair — not a boredom value


def test_a_disabled_thread_logs_a_boredom_once_and_writes_no_file(tmp_path, capsys):
    s = SelfStateThread(tmp_path, enabled=False)
    s.attach_owner_clock(lambda: 7 * H)
    for _ in range(5):
        assert s.mood(now=T0).label == "bored"
    assert not s.mood_history_path.exists() and not s.path.exists()
    assert capsys.readouterr().out.count("quiet for 6 hours") == 1


def test_a_fault_while_journalling_a_boredom_does_not_cost_the_read(tmp_path, monkeypatch):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(s, "_settle_locked", boom)
    assert s.mood(now=_at(hours=7)).label == "bored"


def test_a_settle_writes_decay_only_and_nothing_when_nothing_changed(tmp_path):
    s = SelfStateThread(tmp_path)
    s.mark_owner_turn(now=T0)
    raw = s.path.read_bytes()
    assert s.settle(now=_at(hours=9)).neutral                         # a long quiet: settle is not about boredom
    assert s.path.read_bytes() == raw and _rows(s) == []
    _put(s, 0.9, 0.0)
    raw = s.path.read_bytes()
    s.settle(now=_at(minutes=30))                                      # decayed, same label and level
    assert s.path.read_bytes() == raw and len(_rows(s)) == 1
    view = s.settle(now=_at(hours=2))                                  # a step down: written
    assert (view.label, view.intensity) == ("content", 2) and [r["kind"] for r in _rows(s)] == ["turn", "decay"]


def test_a_settled_step_down_of_an_owner_set_mood_is_still_the_owners(tmp_path):
    """(A mutant the review found alive.) Stored, not only read."""
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 3, now=T0)
    view = s.settle(now=_at(hours=2))
    assert (view.label, view.intensity, view.source) == ("cheerful", 2, "owner")
    assert json.loads(s.path.read_text())["mood"]["source"] == "owner"
    assert SelfStateThread(tmp_path).mood(now=_at(hours=2, minutes=1)).source == "owner"
    gone = s.settle(now=_at(hours=12))                                 # worn off entirely: nobody's
    assert gone.neutral and json.loads(s.path.read_text())["mood"]["source"] == "auto"
    auto = SelfStateThread(tmp_path / "auto")                          # an automatic mood never becomes his
    _put(auto, -0.9, 0.45)
    assert auto.settle(now=_at(hours=2)).source == "auto"


def test_owner_quiet_seconds_reads_the_agents_clock_and_never_writes_it():
    """The fallback clock (`context.last_owner_activity_time`)."""
    from types import SimpleNamespace
    from ghost_agent.selfhood.model import owner_quiet_seconds
    now = dt.datetime(2026, 11, 6, 14, 0)
    ctx = SimpleNamespace(last_owner_activity_time=dt.datetime(2026, 11, 6, 9, 0))
    assert owner_quiet_seconds(ctx, now=now) == 5 * 3600.0
    assert ctx.last_owner_activity_time == dt.datetime(2026, 11, 6, 9, 0)
    for unknown in (SimpleNamespace(), SimpleNamespace(last_owner_activity_time=None),
                    SimpleNamespace(last_owner_activity_time=dt.datetime.min),
                    SimpleNamespace(last_owner_activity_time="09:00"),
                    SimpleNamespace(last_owner_activity_time=dt.datetime(2026, 11, 6, 9, 0, tzinfo=UTC))):
        assert owner_quiet_seconds(unknown, now=now) is None
    ahead = SimpleNamespace(last_owner_activity_time=dt.datetime(2026, 11, 6, 15, 0))
    assert owner_quiet_seconds(ahead, now=now) == 0.0


# ── the one writer ──────────────────────────────────────────────────────────

def test_nudge_clamps_records_and_returns_before_and_after(tmp_path):
    s = SelfStateThread(tmp_path)
    before, after = s.nudge(-5.0, 5.0, "a lot went wrong", M.KIND_TURN, now=T0)
    assert before.neutral and (after.p, after.e) == (-1.0, 1.0)
    assert (after.label, after.intensity, after.source, after.cause) == ("irritable", 3, "auto", "a lot went wrong")
    assert after.set_at == "2026-11-06T09:00:00Z"
    stored = json.loads(s.path.read_text())["mood"]
    assert (stored["p"], stored["e"], stored["updated_at"]) == (-1.0, 1.0, "2026-11-06T09:00:00Z")


def test_a_nudge_applies_on_top_of_the_decayed_pair(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, 0.8, 0.0)
    _b, after = s.nudge(0.1, 0.0, "x", M.KIND_TURN, now=_at(hours=2))
    assert after.p == pytest.approx(0.5)                               # 0.8 halved, then +0.1


@pytest.mark.parametrize("dp,de", [(0.0, 0.0), ("x", 0.1), (None, None)])
def test_a_nudge_that_moves_nothing_writes_nothing(tmp_path, dp, de):
    s = SelfStateThread(tmp_path)
    before, after = s.nudge(dp, de, "nothing", M.KIND_TURN, now=T0)
    assert before == after and not s.path.exists()


def test_recent_keeps_the_last_events_that_moved_the_pair(tmp_path):
    s = SelfStateThread(tmp_path)
    for i in range(MOOD_RECENT_CAUSES + 3):
        s.nudge(0.01, 0.0, f"event {i}", M.KIND_PRAISE, now=_at(minutes=i))
    recent = s.mood(now=_at(minutes=20)).recent
    assert [r["cause"] for r in recent] == [f"event {i}" for i in range(3, MOOD_RECENT_CAUSES + 3)]
    assert MOOD_RECENT_CAUSES == 5 and _rows(s) == []                  # none of them changed the label


# ── the journal ─────────────────────────────────────────────────────────────

def test_a_row_is_written_for_every_change_of_label_or_intensity_and_only_then(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.10, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0)                  # still neutral
    s.nudge(0.15, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP, now=_at(minutes=1))   # → content 1
    s.nudge(0.05, 0.0, "the owner said thanks", M.KIND_PRAISE, now=_at(minutes=2))      # content 1 still
    s.nudge(0.30, 0.0, "a good turn", M.KIND_TURN, now=_at(minutes=3))                  # → content 2
    s.nudge(-0.9, 0.5, "4 tool calls failed in this turn", M.KIND_TURN, now=_at(minutes=4))   # → irritable
    assert [(r["label"], r["intensity"], r["kind"]) for r in _rows(s)] == [
        ("content", 1, "thumb_up"), ("content", 2, "turn"), ("irritable", 2, "turn")]
    row = _rows(s)[0]
    assert set(row) == {"label", "intensity", "p", "e", "at", "cause", "kind"}
    assert (row["e"], row["at"], row["cause"]) == (0.0, "2026-11-06T09:01:00Z", "the owner gave a thumbs-up")
    assert row["p"] == pytest.approx(0.25, abs=2e-3)                   # 0.10 a minute older, plus 0.15


def test_a_mood_that_wore_off_gets_its_row_when_the_next_writer_arrives(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, -0.9, 0.45, cause="6 tool calls failed in this turn")
    s.nudge(0.0, 0.01, "x", M.KIND_PRAISE, now=_at(hours=8))
    rows = _rows(s)
    assert [(r["label"], r["intensity"], r["kind"], r["cause"]) for r in rows] == [
        ("irritable", 3, "turn", "6 tool calls failed in this turn"), ("neutral", 0, "decay", DECAY_CAUSE)]
    assert DECAY_CAUSE == "it wore off"
    # dated when it wore off (0.9 → under 0.16 takes log2(0.9 / 0.16) half-lives ≈ 4 h 59 min),
    # not when the next writer noticed
    assert rows[1]["at"].startswith("2026-11-06T13:59:")


def test_the_time_a_decaying_mood_takes_to_come_down_to_a_level():
    assert M.decay_step_s(0.9, 2) == pytest.approx(M.HALF_LIFE_S * 0.3422, rel=1e-3)     # 0.9 → 0.71
    assert M.decay_step_s(0.9, 1) == pytest.approx(M.HALF_LIFE_S * 1.1344, rel=1e-3)     # 0.9 → 0.41
    assert M.decay_step_s(0.9, 0) == pytest.approx(M.HALF_LIFE_S * 2.4919, rel=1e-3)     # 0.9 → 0.16
    assert M.decay_step_s(0.15, 0) == 0.0 and M.decay_step_s(0.0, 0) == 0.0             # already there
    assert M.decay_step_s(-0.9, 0) == M.decay_step_s(0.9, 0)


def test_what_a_read_shows_does_not_depend_on_who_looked_in_between(tmp_path):
    a, b = SelfStateThread(tmp_path / "a"), SelfStateThread(tmp_path / "b")
    for s in (a, b):
        _put(s, -0.9, 0.45)
    for h in (1, 2, 2.5, 3, 4):                       # b is settled on the way, a is not
        b.settle(now=_at(hours=h))
        va, vb = a.mood(now=_at(hours=h)), b.mood(now=_at(hours=h))
        assert (va.label, va.intensity, va.set_at[:16]) == (vb.label, vb.intensity, vb.set_at[:16]), h


def test_a_pure_read_reports_the_wearing_off_without_writing_it(tmp_path):
    s = SelfStateThread(tmp_path)
    _put(s, -0.9, 0.45)
    v = s.mood(now=_at(hours=3))
    assert (v.label, v.intensity, v.cause) == ("irritable", 1, "it wore off")
    assert v.set_at.startswith("2026-11-06T11:16:")                 # when it came down to 1
    assert len(_rows(s)) == 1 and s.mood(now=_at(hours=8)).cause == ""      # neutral: no cause to name


def test_the_journal_drops_its_oldest_rows_past_the_cap(tmp_path):
    from ghost_agent.selfhood.state import MOOD_HISTORY_KEEP_ROWS, MOOD_HISTORY_MAX_ROWS
    assert (MOOD_HISTORY_MAX_ROWS, MOOD_HISTORY_KEEP_ROWS) == (1200, 1000)
    s = SelfStateThread(tmp_path)
    old = {"label": "content", "intensity": 1, "p": 0.3, "e": 0.0, "at": "2026-11-01T10:00:00Z",
           "cause": "old", "kind": "turn"}
    s.mood_history_path.write_text("".join(json.dumps({**old, "cause": f"old {i}"}) + "\n"
                                           for i in range(MOOD_HISTORY_MAX_ROWS)))
    _put(s, -0.9, 0.0, cause="newest")
    rows = _rows(s)
    assert len(rows) == MOOD_HISTORY_KEEP_ROWS and rows[-1]["cause"] == "newest"
    assert rows[0]["cause"] == f"old {MOOD_HISTORY_MAX_ROWS + 1 - MOOD_HISTORY_KEEP_ROWS}"


def test_the_history_reader_returns_the_newest_rows_oldest_first(tmp_path):
    s = SelfStateThread(tmp_path)
    for i, lab in enumerate(["content", "down", "content", "down"]):
        s.set_owner_mood(lab, M.ANCHOR_DIRECTION[lab], 2, now=_at(minutes=i))
    assert [r["label"] for r in s.mood_history(limit=3)] == ["down", "content", "down"]
    assert s.mood_history(limit=0) == [] and s.mood_history(limit=-1) == []


def test_todays_timeline_is_the_local_days_rows_with_their_local_time(tmp_path):
    s = SelfStateThread(tmp_path)
    tz = dt.timezone(dt.timedelta(hours=2))
    s.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 1,
                     now=dt.datetime(2026, 11, 5, 21, 30, tzinfo=UTC))        # 23:30 local, the day before
    s.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 2,
                     now=dt.datetime(2026, 11, 5, 22, 10, tzinfo=UTC))        # 00:10 local, today
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2,
                     now=dt.datetime(2026, 11, 6, 10, 5, tzinfo=UTC))         # 12:05 local
    today = s.mood_timeline(now=dt.datetime(2026, 11, 6, 12, 0, tzinfo=UTC), tz=tz)
    assert [(r["hhmm"], r["label"], r["intensity"]) for r in today] == [
        ("00:10", "irritable", 2), ("03:58", "neutral", 0), ("12:05", "cheerful", 2)]
    assert s.mood_timeline(now=dt.datetime(2026, 11, 8, 12, 0, tzinfo=UTC), tz=tz) == []


def test_today_is_the_local_day_not_the_utc_day(tmp_path):
    """(A mutant the review found alive.) 23:30 UTC is already tomorrow two
    hours east: the row of 22:10 UTC (00:10 local) is today's, the row of
    21:30 UTC (23:30 local the day before) is not — and the UTC day would
    say the opposite of both."""
    s = SelfStateThread(tmp_path)
    east = dt.timezone(dt.timedelta(hours=2))
    s.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 1, now=dt.datetime(2026, 11, 5, 21, 30, tzinfo=UTC))
    s.set_owner_mood("down", M.ANCHOR_DIRECTION["down"], 2, now=dt.datetime(2026, 11, 5, 22, 10, tzinfo=UTC))
    near_midnight = dt.datetime(2026, 11, 5, 23, 30, tzinfo=UTC)       # 01:30 on the 6th, local
    assert [(r["hhmm"], r["label"]) for r in s.mood_timeline(now=near_midnight, tz=east)] == [("00:10", "down")]
    west = dt.timezone(dt.timedelta(hours=-5))                         # 18:30 on the 5th, local
    assert [r["label"] for r in s.mood_timeline(now=near_midnight, tz=west)] == ["content", "down"]
    assert [r["label"] for r in s.mood_timeline(now=dt.datetime(2026, 11, 6, 0, 30, tzinfo=UTC), tz=west)] == [
        "content", "down"]                                             # the 6th in UTC, still the 5th there


def test_a_byte_that_is_not_utf8_costs_its_own_line_not_the_journal(tmp_path):
    """Review: one bad byte made the `mood` story fail with UnicodeDecodeError,
    lost the diary's timeline and switched the growth cap off."""
    from ghost_agent.selfhood.state import MOOD_HISTORY_KEEP_ROWS, MOOD_HISTORY_MAX_ROWS
    s = SelfStateThread(tmp_path)
    good = {"label": "content", "intensity": 1, "p": 0.3, "e": 0.0, "at": "2026-11-06T08:00:00Z",
            "cause": "the owner gave a thumbs-up", "kind": "thumb_up"}
    with s.mood_history_path.open("wb") as f:
        f.write(json.dumps(good).encode() + b"\n")
        f.write(b'{"label": "down", "intensity": 1, "p": -0.3, "e": 0.0, "at": "2026-11-06T08:05:00Z", '
                b'"cause": "caf\xe9 \xff", "kind": "turn"}\n')            # parses, with replaced characters
        f.write(b'\xff\xfe{"label": "tired"\n')                           # does not parse
        f.write(json.dumps({**good, "label": "calm", "at": "2026-11-06T08:10:00Z"}).encode() + b"\n")
    rows = s.mood_history()
    assert [r["label"] for r in rows] == ["content", "down", "calm"] and "caf" in rows[1]["cause"]
    assert [r["label"] for r in s.mood_timeline(now=T0, tz=UTC)] == ["content", "down", "calm"]
    _put(s, -0.9, 0.0)                                                 # a write beside the bad bytes works
    assert s.mood_history()[-1]["label"] == "down"
    # … and the cap still works, bad bytes carried like any other line
    with s.mood_history_path.open("ab") as f:
        for i in range(MOOD_HISTORY_MAX_ROWS):
            f.write(b'{"label": "content", "intensity": 1, "p": 0.3, "e": 0.0, "at": "2026-11-06T08:20:00Z", '
                    b'"cause": "\xff row", "kind": "turn"}\n')
    s.set_owner_mood("tired", M.ANCHOR_DIRECTION["tired"], 2, now=T0)
    with s.mood_history_path.open("rb") as f:
        assert len(f.readlines()) == MOOD_HISTORY_KEEP_ROWS
    assert s.mood_history(limit=1)[0]["label"] == "tired"


@pytest.mark.parametrize("line", ["7", "null", "true", "[1, 2]", '"content"', "3.5", "[]", '{"label": "content"}',
                                  "{}", "[" * 5000 + "]" * 5000, '{"a": ' * 5000 + "1" + "}" * 5000, "{", ""])
def test_a_line_that_is_json_but_no_row_or_no_json_at_all_is_skipped(tmp_path, line):
    """(Mutants the review found alive.) A number, null, true, a list, a
    string; an object that is no row; a line nested so deep that parsing it
    raises RecursionError — each costs its own line, never the journal."""
    s = SelfStateThread(tmp_path)
    good = {"label": "content", "intensity": 1, "p": 0.3, "e": 0.0, "at": "2026-11-06T08:00:00Z",
            "cause": "the owner gave a thumbs-up", "kind": "thumb_up"}
    s.mood_history_path.write_text(json.dumps(good) + "\n" + line + "\n"
                                   + json.dumps({**good, "label": "calm"}) + "\n")
    assert [r["label"] for r in s.mood_history()] == ["content", "calm"]
    assert [r["label"] for r in s.mood_timeline(now=T0, tz=UTC)] == ["content", "calm"]
    _put(s, -0.9, 0.0)                                                 # and the writer goes on beside it
    assert s.mood_history()[-1]["label"] == "down"


def test_a_disabled_thread_keeps_the_mood_in_memory_and_writes_no_file(tmp_path):
    s = SelfStateThread(tmp_path, enabled=False)
    _put(s, -0.9, 0.0)
    assert s.mood(now=T0).label == "down" and not s.path.exists() and not s.mood_history_path.exists()


# ── the owner's word: anchors, synonyms, manner ─────────────────────────────

@pytest.mark.parametrize("label", M.LABELS)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_each_labels_anchor_reads_back_as_that_label_at_that_intensity(tmp_path, label, n):
    s = SelfStateThread(tmp_path)
    _b, after = s.set_owner_mood(label, M.ANCHOR_DIRECTION[label], n, now=T0)
    assert (after.label, after.intensity, after.source, after.cause) == (label, n, "owner", OWNER_CAUSE)
    again = SelfStateThread(tmp_path).mood(now=T0)                     # from disk, read off the pair
    assert (again.label, again.intensity) == (label, n)
    assert max(abs(after.p), abs(after.e)) == M.ANCHOR_MAGNITUDE[n]


def test_the_anchor_magnitudes_sit_inside_their_bands():
    assert M.ANCHOR_MAGNITUDE == {1: 0.32, 2: 0.60, 3: 0.90}
    for n, m in M.ANCHOR_MAGNITUDE.items():
        assert M.level_for(m) == n and M.level_for(m - 0.1) == n and M.level_for(m + 0.1) == n


@pytest.mark.parametrize("word,label", [
    ("happy", "cheerful"), ("cheery", "cheerful"), ("grumpy", "irritable"), ("annoyed", "irritable"),
    ("cranky", "irritable"), ("relaxed", "calm"), ("sad", "down"), ("low", "down"), ("sleepy", "tired"),
    ("exhausted", "tired"), ("excited", "cheerful"), ("Happy ", "cheerful"), ("TIRED", "tired"),
    ("curious", "curious"), ("bored", "bored"),
])
def test_a_label_or_an_obvious_synonym_resolves(word, label):
    got = M.resolve_label(word)
    assert got is not None and got[0] == label
    assert M.region(*got[1]) == ("cheerful" if label == "curious" else label)


def test_excited_is_cheerful_with_more_energy():
    plain = M.anchor_for(M.resolve_label("cheerful")[1], 2)
    excited = M.anchor_for(M.resolve_label("excited")[1], 2)
    assert excited[1] > plain[1] and excited[0] < plain[0]
    assert excited == pytest.approx((0.45, 0.60)) and plain == pytest.approx((0.60, 0.45))


@pytest.mark.parametrize("word", ["", "  ", "focused", "inquisitive mood", "happy!", "neutral", "clear",
                                  "none", "sarcastic", None, 7, "quietly pleased"])
def test_any_other_word_is_no_mood(word):
    assert M.resolve_label(word) is None


def test_the_clear_words_and_the_manners_are_closed_lists():
    assert M.CLEAR_WORDS == frozenset({"clear", "none", "neutral"})
    assert M.MANNERS == ("sarcastic", "playful", "formal", "terse", "gentle", "dramatic", "evil")
    assert [M.resolve_manner(w) for w in ("Sarcastic", " terse ", "rude", "", None, "neutral")] == [
        "sarcastic", "terse", "", "", "", ""]
    assert not set(M.MANNERS) & set(M.LABELS) and not set(M.SYNONYMS) & set(M.LABELS)


@pytest.mark.parametrize("value,want", [
    (None, 2), (1, 1), (2, 2), (3, 3), ("3", 3), (" 1 ", 1), (2.6, 3), (0, 1), (-4, 1), (9, 3),
    ("strong", 2), (True, 2), (False, 2), ([3], 2), (float("nan"), 2), (float("inf"), 2), ("", 2),
])
def test_the_intensity_is_one_to_three_default_two(value, want):
    assert M.parse_intensity(value) == want and M.DEFAULT_INTENSITY == 2


def test_setting_the_same_mood_again_adds_no_row_and_re_anchors_the_pair(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    _b, after = s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=_at(hours=1))
    assert len(_rows(s)) == 1 and after.set_at == "2026-11-06T09:00:00Z"
    assert after.p == 0.60                                             # back on the anchor, not 0.42
    _b, after = s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 3, now=_at(hours=1))
    assert [(r["label"], r["intensity"], r["kind"]) for r in _rows(s)] == [("calm", 2, "owner"), ("calm", 3, "owner")]


def test_an_owner_set_mood_decays_and_reacts_like_any_mood(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 3, now=T0)
    assert s.mood(now=_at(hours=2)).intensity == 2 and s.mood(now=_at(hours=2)).source == "owner"
    assert s.mood(now=_at(hours=12)).neutral and s.mood(now=_at(hours=12)).source == "auto"
    _b, after = s.nudge(-1.6, 0.0, "4 tool calls failed in this turn", M.KIND_TURN, now=_at(minutes=1))
    assert (after.label, after.source) == ("irritable", "auto")


def test_the_owners_note_rides_the_row_not_the_cause(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("gentle" if False else "calm", M.ANCHOR_DIRECTION["calm"], 1,
                     note="Elena said Bruno is at the vet", now=T0)
    row = _rows(s)[0]
    assert (row["cause"], row["kind"], row["note"]) == (OWNER_CAUSE, "owner", "Elena said Bruno is at the vet")
    assert OWNER_CAUSE == "the owner said so (set_mood)"
    long = SelfStateThread(tmp_path / "long")
    long.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, note="n" * 900, now=T0)
    assert len(_rows(long)[0]["note"]) == 300 and _rows(long)[0]["note"].endswith("…")


def test_neutral_resets_the_pair_and_clears_the_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 3, now=T0)
    s.set_manner("sarcastic", 2, now=T0)
    before, after = s.reset_mood(now=_at(minutes=5))
    assert (before.label, before.manner) == ("irritable", "sarcastic")
    assert after.neutral and (after.p, after.e, after.manner, after.manner_intensity) == (0.0, 0.0, "", 0)
    assert [(r["label"], r["intensity"], r["kind"]) for r in _rows(s)][-1] == ("neutral", 0, "owner")
    assert SelfStateThread(tmp_path).mood(now=_at(minutes=6)) == after


def test_neutral_also_zeroes_what_is_left_under_the_first_threshold(tmp_path):
    """A pair too small to be a mood is still a pair: after "be neutral" a
    single thanks must not tip it over."""
    s = SelfStateThread(tmp_path)
    s.nudge(0.15, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0)
    assert s.mood(now=T0).neutral and s.state.mood.p == 0.15
    before, after = s.reset_mood(now=T0)
    assert before.neutral and (s.state.mood.p, s.state.mood.e) == (0.0, 0.0) and _rows(s) == []
    assert s.nudge(0.10, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0)[1].neutral


def test_nothing_on_file_reads_as_the_empty_view(tmp_path):
    s = SelfStateThread(tmp_path)
    assert s.mood(now=T0) == MoodView() and s.mood() == MoodView()


def test_an_event_leaves_the_manner_where_it_is(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("playful", 3, now=T0)
    _b, after = s.nudge(-0.9, 0.5, "4 tool calls failed in this turn", M.KIND_TURN, now=T0)
    assert (after.label, after.manner, after.manner_intensity) == ("irritable", "playful", 3)
    s.mark_owner_turn(now=T0)
    s.settle(now=_at(hours=9))                                         # … and so do time and a long quiet
    v = s.mood(now=_at(hours=9))
    # §4TS: the manner's level at +9 h depends on where its day ends in THIS
    # machine's local time — read it off the module's own schedule
    want = M.manner_level(3, "2026-11-06T09:00:00Z", now=_at(hours=9))
    assert (v.label, v.manner, v.manner_intensity) == ("bored", "playful" if want else "", want)
    assert v.manner_set_at == ("2026-11-06T09:00:00Z" if want else "")
    # a write keeps a live manner and drops one whose day is over (§4TS closing review)
    assert json.loads(s.path.read_text())["mood"]["manner"] == ("playful" if want else "")


def test_a_manner_leaves_the_mood_alone_and_ends_with_its_day(tmp_path):
    """§4TS: a manner used to stay for good (a month later still "dramatic, 3")."""
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 2, now=T0)
    before, after = s.set_manner("dramatic", 3, now=_at(minutes=1))
    assert (after.label, after.intensity, after.p) == (before.label, before.intensity, before.p)
    assert (after.manner, after.manner_intensity, after.manner_set_at) == ("dramatic", 3, "2026-11-06T09:01:00Z")
    month = s.mood(now=_at(hours=24 * 30))
    assert month.neutral and (month.manner, month.manner_intensity) == ("", 0)
    assert len(_rows(s)) == 1                                          # a manner is not a mood row


def test_clearing_the_manner_clears_only_the_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 2, now=T0)
    s.set_manner("terse", 1, now=T0)
    _b, after = s.set_manner("", 0, now=T0)
    assert (after.manner, after.manner_intensity, after.manner_set_at) == ("", 0, "")
    assert (after.label, after.intensity) == ("content", 2)


def test_a_manner_that_is_no_manner_is_not_stored(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("terse", 2, now=T0)
    _b, after = s.set_manner("rude", 2, now=T0)                        # resolves to "" → clears
    assert after.manner == ""


def test_the_manner_intensity_is_kept_in_range(tmp_path):
    s = SelfStateThread(tmp_path)
    assert s.set_manner("formal", 9, now=T0)[1].manner_intensity == 3
    assert s.set_manner("formal", 0, now=T0)[1].manner_intensity == 1


# ── old files on disk ───────────────────────────────────────────────────────

OLD_STATE = {"schema_version": "v1",
             "mood": {"label": "satisfied", "evidence": "my last 5 verdict-bearing turns all passed",
                      "set_at": "2026-11-05T08:00:00Z", "source": "derived"},
             "open_questions": [{"id": "q1", "text": "why does Bruno circle before lying down"}],
             "last_session_at": "2026-11-05T08:00:00Z"}


@pytest.mark.parametrize("source", ["derived", "self"])
def test_an_old_stored_mood_is_ignored_and_the_state_loads(tmp_path, source):
    (tmp_path / STATE_FILENAME).write_text(json.dumps({**OLD_STATE, "mood": {**OLD_STATE["mood"], "source": source}}))
    s = SelfStateThread(tmp_path)
    assert s.state.mood is None and s.mood(now=T0).neutral
    assert s.state.last_session_at == "2026-11-05T08:00:00Z"           # the rest of the file is kept
    _put(s, 0.3, 0.0)
    assert json.loads(s.path.read_text())["mood"]["label"] == "content"


def test_the_old_history_file_is_kept_and_its_rows_are_skipped(tmp_path):
    old = [{"label": "idle", "evidence": "operator quiet for over 30 minutes", "set_at": "2026-11-05T03:00:00Z",
            "source": "derived"},
           {"label": "stuck", "evidence": "3 of my last 5 verdict-bearing turns failed", "set_at": "t",
            "source": "derived"},
           {"label": "focused", "set_at": "2026-11-05T09:00:00Z"},
           # a label that is also a label today — still an old row: it has no pair
           {"label": "curious", "evidence": "Elena asked about Bruno", "set_at": "2026-11-06T08:00:00Z",
            "source": "self"}]
    (tmp_path / MOOD_HISTORY_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in old))
    s = SelfStateThread(tmp_path)
    assert s.mood_history() == [] and s.mood_timeline(now=T0) == []
    _put(s, 0.3, 0.0)
    assert [r["label"] for r in s.mood_history()] == ["content"]
    assert len(s.mood_history_path.read_text().splitlines()) == 5      # appended to, not replaced


@pytest.mark.parametrize("mood", [
    "irritable", 7, ["irritable"], None, {}, {"p": 0.5}, {"e": 0.5},
    {"label": "irritable", "intensity": 2},
])
def test_a_mood_that_is_not_the_new_shape_is_no_mood(mood):
    assert SelfState.from_dict({"mood": mood}).mood is None


def test_a_hand_edited_mood_costs_the_mood_never_the_state():
    st = SelfState.from_dict({"last_session_at": "2026-11-06T08:00:00Z", "mood": {
        "p": "a lot", "e": 7, "updated_at": 1755000000, "label": "sparkly", "intensity": "2",
        "novelty": "yes", "source": "the cat", "set_at": None, "cause": ["x"], "recent": "none",
        "manner": "rude", "manner_intensity": 99, "manner_set_at": 4}})
    m = st.mood
    assert st.last_session_at == "2026-11-06T08:00:00Z"
    assert (m.p, m.e, m.updated_at, m.label, m.intensity, m.novelty, m.source) == (0.0, 1.0, "", "neutral", 0, False, "auto")
    assert (m.set_at, m.cause, m.recent, m.manner, m.manner_intensity, m.manner_set_at) == ("", "", [], "", 0, "")


def test_a_stored_label_that_is_no_label_is_neutral():
    m = Mood.from_dict({"p": 0.6, "e": 0.0, "label": "sparkly", "intensity": 2})
    assert (m.label, m.intensity, m.p) == ("neutral", 0, 0.6)
    assert Mood.from_dict({"p": 0.6, "e": 0.0, "label": "content", "intensity": 2}).label == "content"


def test_a_stored_label_without_an_intensity_is_neutral_and_recent_is_bounded():
    m = Mood.from_dict({"p": 0.1, "e": 0.0, "label": "content", "intensity": 0,
                        "recent": [{"at": "t", "cause": f"c{i}", "kind": "turn"} for i in range(9)] + ["junk"]})
    assert (m.label, m.intensity) == ("neutral", 0)
    assert [r["cause"] for r in m.recent] == ["c4", "c5", "c6", "c7", "c8"]


def test_a_stored_manner_survives_a_reload_with_a_default_intensity():
    m = Mood.from_dict({"p": 0.0, "e": 0.0, "manner": "formal"})
    assert (m.manner, m.manner_intensity) == ("formal", 2)


def test_the_prefix_survives_a_hand_edited_timestamp(tmp_path):
    """(Ported from the derived-mood tests.) A non-string `updated_at` costs
    the mood line, not the text."""
    (tmp_path / STATE_FILENAME).write_text(json.dumps({
        "mood": {"p": 0.6, "e": 0.0, "updated_at": 1755000000, "label": "content", "intensity": 2},
        "last_session_at": "2026-11-06T08:00:00Z"}))
    s = SelfStateThread(tmp_path)
    text = s.format_as_prefix()
    assert text == "I was last active on 2026-11-06T08:00:00Z."


# ── timestamps (ported from the derived-mood tests: the helpers are kept) ───

def test_parse_iso_utc_handles_z_aware_and_garbage():
    assert M.parse_iso_utc("2026-08-20T10:00:00Z") == dt.datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    assert M.parse_iso_utc("2026-08-20T12:00:00+02:00") == dt.datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    assert M.parse_iso_utc("2026-08-20T10:00:00") == dt.datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    for bad in ("", "   ", "not a date", None, 1755000000, ["2026"], {"t": 1}):
        assert M.parse_iso_utc(bad) is None


def test_age_seconds_and_clamp():
    now = dt.datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
    assert M.age_seconds("2026-08-20T10:00:00Z", now=now) == 7200.0
    assert M.age_seconds("2026-08-20T10:00:00Z", now=now.replace(tzinfo=None)) == 7200.0
    assert M.age_seconds("2026-08-20T14:00:00Z", now=now) == 0.0       # a future stamp is "now"
    assert M.age_seconds("garbage", now=now) is None and M.age_seconds(None, now=now) is None


@pytest.mark.parametrize("x,want", [(0.5, 0.5), (-3, -1.0), (3, 1.0), ("x", 0.0), (None, 0.0),
                                    (float("nan"), 0.0), (float("inf"), 0.0), ("0.25", 0.25)])
def test_clamp(x, want):
    assert M.clamp(x) == want


# ── the removed machinery ───────────────────────────────────────────────────

def test_the_derived_updater_and_everything_that_only_served_it_are_gone():
    import ghost_agent.selfhood as S
    from ghost_agent.selfhood.autobiographical import AutobiographicalMemory
    for name in ("derive_mood", "MoodSignals", "DERIVED_MOOD_VOCAB", "SELF_MOOD_GRACE_HOURS",
                 "MOOD_STALE_AFTER_HOURS", "STREAK_WINDOW", "STREAK_MAX_AGE_DAYS", "IDLE_AFTER_SECONDS",
                 "mood_is_stale", "provenance_phrase", "describe_mood_provenance", "age_phrase"):
        assert not hasattr(M, name), name
        assert not hasattr(S, name), name
    for name in ("update_derived_mood", "_maybe_retire_acute_mood"):
        assert not hasattr(SelfModel, name), name
    for name in ("set_mood", "clear_mood"):
        assert not hasattr(SelfStateThread, name), name
    assert not hasattr(AutobiographicalMemory, "recent_verdicts")


def test_a_disabled_self_model_reads_neutral_and_moves_nothing(tmp_path):
    sm = SelfModel(root=tmp_path, enabled=False)
    assert sm.mood() == MoodView() and sm.mood().neutral
    assert sm.note_owner_turn(M.TurnSignals(tool_calls=5, failures=5)) is None
    assert sm.note_owner_reaction(M.KIND_THUMB_UP) is None
    assert sm.owner_arrived() is None and sm.owner_turn_ended() is None
    sm.attach_owner_clock(lambda: 9 * 3600.0)                          # nothing to attach to: no raise
    assert sm.stats() == {"enabled": False}
    assert not any(Path(tmp_path).iterdir())


def test_a_store_fault_never_raises_out_of_the_facade(tmp_path, monkeypatch):
    sm = SelfModel(root=tmp_path)

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(sm.state, "nudge", boom)
    monkeypatch.setattr(sm.state, "owner_arrived", boom)
    monkeypatch.setattr(sm.state, "mark_owner_turn", boom)
    monkeypatch.setattr(sm.state, "mood", boom)
    assert sm.note_owner_reaction(M.KIND_THUMB_UP) is None
    assert sm.note_owner_turn(M.TurnSignals(tool_calls=5, failures=2)) is None
    assert sm.owner_arrived() is None and sm.owner_turn_ended() is None
    assert sm.mood() == MoodView()


def test_the_stats_carry_the_mood_as_it_is_now(tmp_path):
    sm = SelfModel(root=tmp_path)
    st = sm.stats()
    assert (st["last_mood"], st["last_mood_intensity"], st["last_mood_source"], st["last_mood_cause"],
            st["manner"], st["manner_intensity"]) == ("", 0, "auto", "", "", 0)
    assert "last_mood_evidence" not in st
    sm.note_owner_reaction(M.KIND_THUMB_DOWN)
    sm.state.set_manner("gentle", 1)
    st = sm.stats()
    assert (st["last_mood"], st["last_mood_intensity"], st["last_mood_source"], st["last_mood_cause"],
            st["manner"], st["manner_intensity"]) == ("down", 1, "auto", "the owner gave a thumbs-down", "gentle", 1)
    assert st["last_mood_set_at"].endswith("Z")


# ── the diary reads the mood from the JOURNAL ───────────────────────────────

def _diary(tmp_path):
    from ghost_agent.selfhood.narrative import NarrativeSummariser
    return NarrativeSummariser(tmp_path, critique_fn=None)


def test_the_diarys_state_block_carries_the_days_timeline(tmp_path):
    s = SelfStateThread(tmp_path)
    n = _diary(tmp_path)
    assert n._format_state_block(None) == "(no persisted state)"
    assert n._format_state_block(s) == "(nothing on file)"
    s.nudge(0.3, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP)
    s.nudge(-0.9, 0.5, "4 tool calls failed in this turn", M.KIND_TURN)
    assert n._format_state_block(s).splitlines() == [
        "My mood today: content 1 (the owner gave a thumbs-up) → irritable 2 (4 tool calls failed in this turn)."]


def test_the_diarys_block_does_not_move_with_the_clock_or_with_a_boredom_level(tmp_path):
    """Review: three diary calls for ONE quiet stretch — the block changed
    when a read showed bored 1, then bored 2. It is read off the journal's
    own events: the same text while the mood decays, while the quiet reaches
    each level, and after the owner returns. The one thing that may move it
    is the "it wore off" row, a thing that happened to the mood: the first
    read that sees the boredom writes it, dated when the mood wore off, so
    it is on today's page or yesterday's depending on the hour this runs."""
    wore_off = " → neutral (it wore off)"
    s = SelfStateThread(tmp_path)
    n = _diary(tmp_path)
    s.touch_session()
    now = dt.datetime.now(UTC)
    s.mark_owner_turn(now=now)
    s.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 2, now=now)
    s.state.mood.updated_at = (now - dt.timedelta(hours=20)).replace(tzinfo=None).isoformat() + "Z"
    s.state.last_owner_turn_at = s.state.mood.updated_at
    first = n._format_state_block(s)
    assert "content 2" in first
    assert "ago" not in first and "bored" not in first and "My mood:" not in first
    assert s.mood().label == "bored" and s.mood().intensity == 2       # a reader sees bored 2 right now …
    assert {r["kind"] for r in s.mood_history()} >= {"quiet"}          # … and its rows are in the journal
    settled, query = n._format_state_block(s), n._derive_recall_query(s)
    assert settled.replace(wore_off, "") == first                      # only that row can be new
    assert s.mood().label == "bored"                                   # another read of the same quiet
    assert n._format_state_block(s) == settled and n._derive_recall_query(s) == query
    s.owner_arrived()
    assert s.mood_history()[-1]["kind"] == "quiet_end"
    assert n._format_state_block(s) == settled and n._derive_recall_query(s) == query


def test_the_diarys_timeline_is_bounded(tmp_path):
    from ghost_agent.selfhood import narrative as N
    s = SelfStateThread(tmp_path)
    for i in range(12):
        lab = "content" if i % 2 else "down"
        s.set_owner_mood(lab, M.ANCHOR_DIRECTION[lab], 2)
    line = N.NarrativeSummariser(tmp_path, critique_fn=None)._format_state_block(s).splitlines()[-1]
    assert line.count("→") == N._MOOD_TIMELINE_STEPS - 1 == 7


def test_the_diarys_recall_query_is_the_label_of_the_days_last_step(tmp_path):
    """(Ported: label not evidence.) Read off the journal: nothing for a day
    without a step, nothing once the last step is "neutral", never "bored"."""
    s = SelfStateThread(tmp_path)
    n = _diary(tmp_path)
    assert n._derive_recall_query(None) == "" and n._derive_recall_query(s) == ""
    s.nudge(-0.9, 0.5, "4 tool calls failed in this turn", M.KIND_TURN)
    assert n._derive_recall_query(s) == "irritable"
    s.reset_mood()
    assert n._derive_recall_query(s) == ""
    s.mark_owner_turn(now=dt.datetime.now(UTC) - dt.timedelta(hours=7))
    assert s.mood().label == "bored" and n._derive_recall_query(s) == ""
