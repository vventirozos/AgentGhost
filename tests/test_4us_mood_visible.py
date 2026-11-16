"""§4US — the character system's second live look (2026-11-16), and its three fixes.

Operator: "proceed with 1 and 2. for 3. this is my own personal AI so i don't do a lot of requests each
day. so the mood should be visible even if there is not a ton of requests."

1. The correction signal was dead. The judge on Nova answers ``{"correct": true}`` (the key without its
   "s") on every call, so every ruling read ``None``: no correction reached the mood since at least 10-07.
   The phrase gate also missed "well you are wrong", "you didn't generate an image" and "i don't like
   it… i told you".
2. A three-turn day pulled humour as hard as a thirty-turn day.
3. With a few requests a day the mood was neutral at 10 of 11 owner replies. The half-life is now 4 h
   (was 2 h), and the owner coming back after a long quiet is a small lift.
"""
from __future__ import annotations

import datetime as dt

import pytest

from ghost_agent.distill.user_correction import has_correction_phrase, parse_adjudication
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import mood as M

T0 = dt.datetime(2026, 11, 16, 9, 0, tzinfo=dt.timezone.utc)


# ── 1. corrections ───────────────────────────────────────────────────────────

def test_the_judges_real_answer_is_a_ruling():
    """The live answer, verbatim (8 of 8 live cases had this shape)."""
    assert parse_adjudication('{"correct": true}') is True
    assert parse_adjudication('{"correct": false}') is False
    assert parse_adjudication('{"is_correction": true}') is True
    assert parse_adjudication('{"user_correct_assistant": true}') is True    # its answer to that key, live
    assert parse_adjudication('{"corrects": true}') is True             # what the prompt asks for


@pytest.mark.parametrize("junk", ['{"correct": true, "sure": false}', '{"answer": true}',
                                  '{"correct": "yes"}', '[true]'])
def test_an_answer_that_is_not_one_boolean_under_a_correct_key_has_not_ruled(junk):
    assert parse_adjudication(junk) is None


@pytest.mark.parametrize("text", [
    "well you are wrong. its only 160kn away",                          # 11-15, live
    "you didn't generate an image",                                     # 10-11, live
    "i don't like it. rewrite it from scratch, i told you to make it look awesome",   # 10-18, live
    "you're wrong", "that is wrong", "but that's not right", "it still doesn't work",
    "κάνεις λάθος", "δεν το έκανες",
])
def test_the_gate_lets_the_direct_corrections_reach_the_judge(text):
    assert has_correction_phrase(text)


@pytest.mark.parametrize("text", ["well done", "good morning ghost", "so what did you find?",
                                  "you did great!", "that is perfect", "it works now, thanks"])
def test_the_gate_still_ignores_ordinary_chat(text):
    assert not has_correction_phrase(text)


@pytest.mark.asyncio
async def test_a_live_shaped_ruling_reaches_the_mood_as_a_correction():
    """The judge's real answer, through `_adjudicate_correction`: it rules (before §4US: None)."""
    from ghost_agent.core.agent import GhostAgent

    class _V:
        llm_client = object()

        async def _call_llm(self, *a, **k):
            return {"choices": [{"message": {"content": '{"correct": true}'}}]}

    a = GhostAgent.__new__(GhostAgent)
    a.context = type("C", (), {"verifier": _V(), "trajectory_collector": None})()
    msgs = [{"role": "user", "content": "how far is tripoli from athens by car?"},
            {"role": "assistant", "content": "About 220 km, roughly 2 h 15 m."},
            {"role": "user", "content": "well you are wrong. its only 160kn away"}]
    assert await a._adjudicate_correction(msgs, msgs[-1]["content"]) is True


# ── 2. humour weighs a day by its turns ─────────────────────────────────────

def _day(date, clean, trouble):
    return {"date": date, "own": {"clean_turns": clean, "trouble": trouble}}


def test_a_day_counts_for_humour_in_proportion_to_its_turns():
    assert C._day_weight(_day("d", 2, 1), "humour") == pytest.approx(3 / C.WORK_FULL_TURNS)
    assert C._day_weight(_day("d", 20, 5), "humour") == 1.0
    assert C._day_weight(_day("d", 2, 1), "warmth") == 1.0 == C._day_weight(_day("d", 2, 1), "directness")
    assert C._day_weight({"date": "d", "own": "x"}, "humour") == 0.0


def test_a_thin_bad_day_moves_humour_less_than_a_full_one():
    base = [_day(f"2026-11-0{i}", 9, 1) if i % 2 else _day(f"2026-11-0{i}", 8, 2) for i in range(1, 8)]
    thin = C.day_shifts(base + [_day("2026-11-08", 2, 1)])["2026-11-08"]["humour"]
    full = C.day_shifts(base + [_day("2026-11-08", 20, 10)])["2026-11-08"]["humour"]
    assert full < thin < 0 and abs(thin) <= abs(full) * 0.5


# ── 3. a mood a few visits a day can hear ───────────────────────────────────

def test_the_half_life_is_four_hours():
    assert M.HALF_LIFE_S == 4 * 3600.0
    assert M.decay(0.8, 4 * 3600) == pytest.approx(0.4)


@pytest.mark.parametrize("quiet,expect", [(None, None), ("x", None), (float("nan"), None), (float("inf"), None),
                                          (5.9 * 3600, None), (6 * 3600, 6), (13.5 * 3600, 13)])
def test_the_return_lift_needs_a_long_quiet(quiet, expect):
    ev = M.return_event(quiet)
    if expect is None:
        assert ev is None
    else:
        assert (ev.dp, ev.de, ev.kind) == (*M.EV_RETURN, M.KIND_RETURN)
        assert ev.cause == f"the owner came back after {expect} hours"


def test_coming_back_after_a_night_is_heard_as_cheerful(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.state.mark_owner_turn(now=T0 - dt.timedelta(hours=9))
    sm.owner_arrived(now=T0)
    v = sm.state.mood(now=T0)
    assert (v.label, v.intensity) == ("cheerful", 1)
    assert sm.state.state.mood.cause == "the owner came back after 9 hours"
    sm.owner_arrived(now=T0 + dt.timedelta(minutes=5))                  # the next message is no second lift
    assert [e["kind"] for e in sm.state.mood_events()].count(M.KIND_RETURN) == 1


def test_a_short_break_is_no_lift(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.state.mark_owner_turn(now=T0 - dt.timedelta(hours=2))
    sm.owner_arrived(now=T0)
    assert sm.state.state.mood is None


def test_the_return_is_felt_by_warmth():
    ev = M.return_event(7 * 3600)
    assert C.lean(ev, {"warmth": 2, "humour": 0, "directness": 0}).dp == pytest.approx(0.30)
    assert C.lean(ev, {"warmth": -2, "humour": 0, "directness": 0}).dp == pytest.approx(0.10)


# ── review r1 (§4US) ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("ans", ['{"is_correct": true}', '{"correctness": true}', '{"correct_answer": true}',
                                 '{"_correct": true}', '{"corrects": "yes", "correct": true}'])
def test_keys_that_say_the_assistant_was_right_are_no_ruling(ans):
    assert parse_adjudication(ans) is None


@pytest.mark.parametrize("prev,cur", [
    ("search the weather in athens", "ok no worries, then search the weather in athens tomorrow"),
    ("make an image of a red car", "you didn't have to make a red car image, nice"),
])
def test_without_a_judge_the_new_openers_promote_nothing(prev, cur):
    """r1: a judge that timed out (contradicts=None) falls back to phrase + rephrase; the §4US openers are
    for the judge only."""
    from ghost_agent.distill.user_correction import classify_user_correction
    assert not classify_user_correction(prev_user_request=prev, prev_assistant_response="Done.",
                                        current_user_text=cur, contradicts=None).is_correction
    assert has_correction_phrase(cur)                                   # … the judge is still asked


def test_an_owner_job_through_the_night_does_not_hide_that_the_owner_was_away(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.state.mark_owner_turn(now=T0 - dt.timedelta(hours=10))
    sm.state.mark_owner_job(now=T0 - dt.timedelta(minutes=20))
    sm.owner_arrived(now=T0)
    assert sm.state.state.mood.cause == "the owner came back after 10 hours"


def test_the_log_line_says_a_thin_day_counts_less():
    base = [_day(f"2026-11-0{i}", 9, 1) if i % 2 else _day(f"2026-11-0{i}", 8, 2) for i in range(1, 8)]
    line = C.pull_line(base + [_day("2026-11-08", 2, 1)], "2026-11-08")
    assert "(a thin day: counts 0.3)" in line and "humour" in line
