"""§4UP — the character system's evaluation fixes (2026-11-14, operator: "look at the whole character system …
end to end. the data it generates, the signal strength, everything and evaluate it." → "proceed with everything").

What the live data of 11-09..11-14 showed, and what each test below pins:

1. a clean turn of 1–2 calls earned its full reward, so every small turn lifted the mood ("content" on 49 of 79
   owner replies): it earns a quarter now (``mood.CLEAN_TURN_SMALL_SHARE``);
2. long work drained the energy and the map called pure low energy "bored" (a 173-min, 60-call turn → "bored 1"):
   that axis reads "tired"; "bored" is the quiet's and the owner's word only;
3. greetings scored as upbeat tone ("good morning ghost!" 0.93): with an emotion reading, a tone counts only with
   an emotion on its side, and an unhappy tone an annoyed emotion backs needs 0.60 (none had reached 0.85);
4. warmth was a COUNT of warm messages (three days at the 3.0 cap): it is a share of the messages read;
5. the daily rule was a lead of DAYS, which a steady signal at its ceiling wins (directness pulled up on 8 of 9
   judged days): it reads the size of each day's shift, √n-scaled, with a spread floor;
6. the diary was rewritten daily on the main model and read by nothing: it is off unless switched on.
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import narrative as N
from ghost_agent.selfhood import sentiment as S
from tests.test_4tc_mood_hooks import owner  # noqa: F401 — the fixture (an owner request)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GHOST_SELFHOOD_NARRATIVE", raising=False)
    S.BREAKER.reset()
    yield
    S.BREAKER.reset()


# ── 1. small turns ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("calls,dp", [(1, 0.015), (2, 0.02), (3, 0.10), (6, 0.16)])
def test_a_small_clean_turn_earns_a_quarter_and_real_work_its_full_reward(calls, dp):
    (ev,) = M.turn_events(M.TurnSignals(tool_calls=calls))
    assert ev.dp == pytest.approx(dp)


def test_a_run_of_small_turns_alone_stays_under_the_first_level():
    """Ten 1-call turns ten minutes apart: the decayed sum stays under level 1 (it was content 2 at the full
    share: 0.06 a turn, half-life 2 h)."""
    p = 0.0
    for _ in range(10):
        p = M.decay(p, 600) + M.turn_events(M.TurnSignals(tool_calls=1))[0].dp
    assert M.read_off(p, 0.0)[1] == 0
    full = 0.0
    for _ in range(10):
        full = M.decay(full, 600) + 0.06
    assert M.read_off(full, 0.0)[1] == 2                      # what the old share made of the same hour


# ── 2. effort is tired, not bored ───────────────────────────────────────────

def test_the_turn_the_evaluation_found_now_reads_tired():
    """The live 11-14 13:56 turn: 59 calls, 3 failed, 173 minutes (journalled +0.062, −0.24 → "bored 1")."""
    ev = M.merge_events(M.turn_events(M.TurnSignals(tool_calls=59, failures=3, duration_s=173 * 60)))
    assert (round(ev.dp, 3), round(ev.de, 3)) == (0.062, -0.24)
    assert M.read_off(ev.dp, ev.de)[:2] == ("tired", 1)


@pytest.mark.parametrize("p,e,label", [
    (0.0, -0.3, "tired"), (0.05, -0.3, "tired"), (-0.05, -0.3, "tired"),   # the low-energy axis
    (-0.3, -0.3, "tired"),                                                 # the unpleasant, low-energy quarter
    (0.3, -0.3, "calm"), (0.0, 0.3, "restless"), (0.3, 0.0, "content"), (-0.3, 0.0, "down"),
])
def test_the_map_has_no_bored_region(p, e, label):
    assert M.region(p, e) == label


def test_the_owners_bored_still_holds_and_a_tired_mood_keeps_its_name_on_the_axis():
    # the owner's "bored" anchor sits on the axis: held, it reads bored as it decays
    p, e = M.anchor_for(M.ANCHOR_DIRECTION["bored"], 2)
    assert M.read_off(p, e, held_label="bored", held_intensity=2)[0] == "bored"
    assert M.read_off(p / 3, e / 3, held_label="bored", held_intensity=2)[0] == "bored"
    # a tired mood that drifts from the quarter onto the axis is still tired (one region, the widened margin)
    assert M.held_region(-0.05, -0.3, "tired") == "tired"
    assert M.held_region(-0.3, -0.25, "tired") == "tired"
    assert not M._still_in(0.3, -0.3, "tired")               # pleasant and low energy is calm, not tired
    assert not M._still_in(-0.3, 0.05, "tired")              # any energy at all is not tired


@pytest.mark.parametrize("p,e,held,label", [
    # review r1 (mutants M1/M16 survived): the widened edges of BOTH halves of "tired"
    (0.14, -0.3, "tired", "tired"), (0.14, -0.3, "", "calm"),          # axis edge, pleasant side: ratio .47
    (0.16, -0.3, "tired", "calm"),                                      # ratio .53: clearly past
    (-0.3, -0.1, "tired", "tired"), (-0.3, -0.1, "", "down"),           # quarter edge: ratio .33
    (-0.3, -0.08, "tired", "down"),                                     # ratio .27: clearly past
    (0.14, -0.3, "calm", "calm"), (0.08, -0.3, "calm", "tired"),        # from calm: .47 raw calm; .27 past
])
def test_tired_is_held_by_the_margin_on_both_of_its_edges(p, e, held, label):
    assert M.held_region(p, e, held) == label


def test_the_quiet_is_still_what_makes_bored():
    assert M.boredom_level(7 * 3600) == 1 and M.boredom_level(13 * 3600) == 2 and M.boredom_level(3600) == 0


@pytest.mark.parametrize("rest,face", [((0.0, -0.12), "calm"), ((0.06, -0.12), "calm"), ((0.12, -0.06), "calm"),
                                       ((0.12, 0.06), "cheerful"), ((0.0, 0.0), "neutral")])
def test_a_resting_point_on_the_low_energy_axis_is_calm_on_the_face(rest, face):
    assert M.rest_label(rest) == face and M.face_label("neutral", 0, rest) == face


# ── 3. the tone needs an emotion on its side ───────────────────────────────

@pytest.mark.parametrize("label,score,emotions,moves", [
    ("positive", 0.93, {}, False),                                   # "good morning ghost!" — politeness
    ("positive", 0.93, {"curiosity": 0.7}, False),                   # a question is not warmth
    ("positive", 0.93, {"admiration": 0.86}, True),                  # "awesome, now add minesweeper"
    ("positive", 0.80, {"gratitude": 0.99}, False),                  # still under its own 0.85
    ("positive", 0.93, {"admiration": 0.45}, False),                 # under the count's cut
    ("positive", 0.93, None, True),                                  # no emotion model: the old rule
    ("negative", 0.66, {}, False),                                   # "delete project cd17…" read 0.66
    ("negative", 0.66, {"annoyance": 0.7}, True),                    # annoyed: the bar is 0.60
    ("negative", 0.55, {"annoyance": 0.9}, False),
    ("negative", 0.90, {"sadness": 0.9}, True),                      # review r1: the owner's bad news still lowers it
    ("negative", 0.66, {"fear": 0.6}, True),                         # … backed, at the 0.60 bar
    ("negative", 0.66, {"curiosity": 0.9}, False),                   # a question is no bad news
    ("positive", 0.93, {"admiration": 0.5}, True),                   # exactly the cut counts (mutant: > for >=)
    ("negative", 0.66, {"annoyance": 0.5}, True),
    ("negative", 0.90, None, True),                                  # no emotion model: the old rule
    ("negative", 0.70, None, False),
    ("neutral", 0.99, {"admiration": 0.9}, False),
])
def test_a_tone_counts_only_with_an_emotion_on_its_side(label, score, emotions, moves):
    assert (M.sentiment_event(label, score, emotions) is not None) == moves


def test_the_emotion_sides_are_one_list():
    assert S.WARM_EMOTIONS is M.SENTIMENT_WARM and S.COLD_EMOTIONS is M.SENTIMENT_COLD
    # the mood's unhappy side is wider than the warmth dial's annoyed one: bad news is not annoyance with Ghost
    assert M.SENTIMENT_COLD < M.SENTIMENT_UNHAPPY and "sadness" in M.SENTIMENT_UNHAPPY - M.SENTIMENT_COLD
    assert not (M.SENTIMENT_UNHAPPY & M.SENTIMENT_WARM)
    assert S.EMOTION_COUNT_MIN == M.SENTIMENT_EMOTION_MIN
    assert not (M.SENTIMENT_WARM & M.SENTIMENT_COLD) and "curiosity" not in M.SENTIMENT_WARM


def _model(tmp_path):
    from ghost_agent.selfhood import SelfModel
    return SelfModel(tmp_path / "selfhood")


def test_a_greeting_is_held_as_politeness_and_says_so(tmp_path):
    sm = _model(tmp_path)
    nudge, held = S._feed_mood({"label": "positive", "score": 0.93, "emotions": {"neutral": 0.9}}, sm, False,
                               None, "m1", "good morning ghost!")
    assert (nudge, held) == (0.0, S.HELD_UNBACKED)
    assert sm.mood().p == 0.0
    line = S.rating_line({"label": "positive", "score": 0.93, "probs": {"positive": 0.93}, "emotions": {}},
                         nudge, held)
    assert "mood unchanged: no emotion backs the tone" in line


def test_an_annoyed_message_under_the_old_bar_moves_the_mood_down(tmp_path):
    """The emotions reach the SelfModel too: it rebuilds the event (an earlier draft passed them to the gate
    only, and the model's own rebuild at the 0.85 bar dropped the move as "no mood to move")."""
    sm = _model(tmp_path)
    nudge, held = S._feed_mood({"label": "negative", "score": 0.66, "emotions": {"annoyance": 0.8}}, sm, False,
                               None, "m1", "this keeps happening, I am fed up")
    assert held == "" and nudge == pytest.approx(M.EV_SENTIMENT_NEGATIVE[0])
    assert sm.mood().p == pytest.approx(M.EV_SENTIMENT_NEGATIVE[0])


def test_an_unhappy_score_without_annoyance_is_below_its_bar(tmp_path):
    sm = _model(tmp_path)
    assert S._feed_mood({"label": "negative", "score": 0.66, "emotions": {}}, sm, False, None, "m1",
                        "the problem is that most") == (0.0, "below 0.85")


# ── 4. warmth is a share ─────────────────────────────────────────────────────

def _day(date, **kw):
    rec = {"date": date, "thumbs_up": 0, "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0,
           "tone_down": 0, "msgs_rated": 0, "msgs_warm": 0, "msgs_cold": 0, "msgs_thanks": 0,
           "own": {"trouble": 0, "clean_turns": 0}, "shades": {}}
    rec.update(kw)
    return rec


def test_warmth_reads_how_the_owner_talked_not_how_much():
    """The live 11-13 day (15 read, 6 warm) and a chatty day with the same share read the same; a quiet day of
    two warm messages says nothing."""
    assert C.day_signals(_day("d", msgs_rated=15, msgs_warm=6))["warmth"] == pytest.approx(0.4)
    assert C.day_signals(_day("d", msgs_rated=30, msgs_warm=12))["warmth"] == pytest.approx(0.4)
    assert C.day_signals(_day("d", msgs_rated=5, msgs_warm=5))["warmth"] == 1.0
    assert C.day_signals(_day("d", msgs_rated=2, msgs_warm=2))["warmth"] is None
    assert C.day_signals(_day("d", msgs_rated=4, msgs_cold=4))["warmth"] == -2.0


# ── 5. the daily rule reads the size of each day's shift ────────────────────

def _dated(recs, start=datetime.date(2026, 1, 1)):
    return [dict(r, date=(start + datetime.timedelta(days=i)).isoformat()) for i, r in enumerate(recs)]


def _replay(days, dials=None):
    """The daily rule run once per day, as the idle window does — the moves it made."""
    state = {"dials": dict(dials or {"warmth": 0, "humour": 0, "directness": 0}), "pins": [], "changed": {}}
    moves = []
    ordered = C._by_date(days)
    for i in range(len(ordered)):
        for m in C.propose(ordered[:i + 1], state, [], ordered[i]["date"]):
            state["dials"][m["dial"]] = m["to"]
            state["changed"][m["dial"]] = ordered[i]["date"]
            moves.append((ordered[i]["date"], m["dial"], m["to"]))
    return moves


def _thumb_days(n, bad_every=0, bad_from=0):
    """Every day 2 thumbs up; every ``bad_every``-th day (from day ``bad_from``) also a thumbs-down."""
    return _dated([_day("d", thumbs_up=2, thumbs_down=1 if bad_every and i >= bad_from and i % bad_every == 0 else 0)
                   for i in range(n)])


def test_a_steady_signal_at_its_ceiling_moves_no_dial():
    """The evaluation's ratchet: a thumbs-down one day in four, every other day all up. An all-up day sits 0.58 of
    a spread above that norm, a bad day 1.73 below — so counting DAYS (the old rule: 3 a step back, 4 away) read
    three pulls up for each one down, on a signal that never changed; the size of the shifts sums to 0."""
    days = _thumb_days(120, bad_every=4)
    assert _replay(days) == []
    pulls = C.day_pulls(days)
    last14 = [pulls.get(d["date"], {}).get("directness", 0) for d in C._by_date(days)[-14:]]
    assert last14.count(1) - last14.count(-1) >= 4                     # the old lead of days: a step away from 0


def test_a_real_change_moves_the_dial_and_the_story_names_its_size():
    """All up for six weeks, then a thumbs-down every other day: directness steps down."""
    days = _thumb_days(42) + _thumb_days(10, bad_every=2)
    days = _dated(days)
    moves = _replay(days)
    assert moves and moves[0][1:] == ("directness", -1)
    st = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    first = next(m for m in (C.propose(days[:i + 1], st, [], days[i]["date"]) for i in range(len(days))) if m)[0]
    assert first["shift"] <= -C.SHIFT_OUT and first["days_read"] >= C.SHIFT_MIN_DAYS
    assert "of a day's spread" in first["why"] and first["days_down"]


def test_a_norm_of_identical_days_has_the_dials_floor_for_a_spread():
    days = _dated([_day("d", thumbs_up=2)] * 6 + [_day("d", thumbs_up=1, thumbs_down=1)])
    z = C.day_shifts(days)[days[-1]["date"]]["directness"]
    assert z == -C.SHIFT_CLAMP                                # (−1 − 1) / 0.4 = −5, clamped
    days = _dated([_day("d", thumbs_up=2)] * 6 + [_day("d", thumbs_up=3, thumbs_down=0)])
    assert C.day_shifts(days)[days[-1]["date"]]["directness"] == 0.0     # the same best day is an ordinary day
    assert set(C.SPREAD_FLOOR) == set(C.DIALS) and all(
        (min(v.values()) if isinstance(v, dict) else v) > 0 for v in C.SPREAD_FLOOR.values())


def _after_change(seq, dial_at=1, pre=25):
    """Review r1's scenario: ``pre`` all-up days, the dial changed on the last of them, then ``seq`` (u = all
    up, d = one thumbs-down, m = 2 up and 1 down). The first move of directness, or None."""
    recs = [_day("d", thumbs_up=2)] * pre + [
        _day("d", thumbs_up=2) if k == "u" else _day("d", thumbs_down=1) if k == "d" else
        _day("d", thumbs_up=2, thumbs_down=1) for k in seq]
    days = _dated(recs)
    st = {"dials": {"warmth": 0, "humour": 0, "directness": dial_at}, "pins": [],
          "changed": {"directness": days[pre - 1]["date"]}}
    for k in range(pre, len(days)):
        mv = [m for m in C.propose(days[:k + 1], st, [], days[k]["date"]) if m["dial"] == "directness"]
        if mv:
            return mv[0]["from"], mv[0]["to"]
    return None


@pytest.mark.parametrize("seq", ["uud", "duu", "udu", "uuud", "uuuud", "m", "uum", "d"])
def test_one_bad_day_never_undoes_an_earned_step(seq):
    """Review r1 (MAJOR): with the shift clamped at 3, "up, up, down" moved directness +1 → 0 and "down, up, up"
    did not — one day carried the move, and its place in the order decided it."""
    assert _after_change(seq) is None


def test_one_day_alone_can_never_reach_a_move():
    """The cap is what keeps ONE day under the bar, whatever the days around it read 0: at the fewest days read,
    one capped day is short of the step back (at 3 it was 1.73 ≥ 1.5)."""
    import math
    assert C.SHIFT_CLAMP / math.sqrt(C.SHIFT_MIN_DAYS) < C.SHIFT_IN < C.SHIFT_OUT


def test_a_move_needs_two_days_that_pull_clearly(monkeypatch):
    """… and the second guard, alone: one extreme day plus ordinary days that LEAN the same way (inside the band,
    so not pulling) add up past the bar — still no move with one clear day; with two, it moves."""
    days = _dated([_day("d")] * 3)
    st = {"dials": {"warmth": 0, "humour": 0, "directness": 1}, "pins": [], "changed": {}}
    monkeypatch.setattr(C, "day_shifts", lambda ds: {d["date"]: {"directness": z} for d, z in
                                                     zip(C._by_date(ds), (-2.5, -0.45, -0.45))})
    monkeypatch.setattr(C, "day_pulls", lambda ds: {C._by_date(ds)[0]["date"]: {"directness": -1}})
    assert C.propose(days, st, [], days[-1]["date"]) == []                   # (−3.4)/√3 = −1.96, one clear day
    monkeypatch.setattr(C, "day_pulls", lambda ds: {d["date"]: {"directness": -1} for d in C._by_date(ds)[:2]})
    assert [(m["from"], m["to"]) for m in C.propose(days, st, [], days[-1]["date"])] == [(1, 0)]


def test_two_bad_days_close_together_are_a_change():
    assert _after_change("udud") == (1, 0)
    assert _after_change("duuuuud", dial_at=0) is None             # two, a week apart, from 0: not yet away


def test_too_few_days_read_move_nothing():
    days = _dated([_day("d", thumbs_up=2)] * 6 + [_day("d", thumbs_down=2)] * 2)
    assert C.propose(days, {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {
        "directness": days[5]["date"]}}, [], days[-1]["date"]) == []                   # 2 days read < 3


def test_a_nan_shift_is_no_reading(monkeypatch):
    days = _dated([_day("d")] * 4)
    monkeypatch.setattr(C, "day_shifts", lambda ds: {d["date"]: {"humour": float("nan")} for d in ds})
    monkeypatch.setattr(C, "day_pulls", lambda ds: {})
    assert C.propose(days, {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}},
                     [], days[-1]["date"]) == []


# ── 6. the diary is off unless switched on ──────────────────────────────────

def test_the_diary_is_off_by_default(monkeypatch):
    assert N.enabled() is False
    monkeypatch.setenv("GHOST_SELFHOOD_NARRATIVE", "1")
    assert N.enabled() is True
    monkeypatch.setenv("GHOST_SELFHOOD_NARRATIVE", "true")
    assert N.enabled() is False


async def test_the_idle_window_writes_no_diary_while_it_is_off():
    from ghost_agent.core.agent import GhostAgent
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=1200)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.self_narrative_cooldown = None
    ctx.frontier_tracker = None
    ctx.trajectory_collector = None
    ctx.self_model = MagicMock(enabled=True)
    ctx.self_model.consolidate_narrative = AsyncMock(return_value="A diary entry.")
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    ctx.self_model.consolidate_narrative.assert_not_called()


def _introspect_model(tmp_path, text):
    from ghost_agent.selfhood import SelfModel
    sm = SelfModel(tmp_path / "selfhood")
    sm.narrative.path.parent.mkdir(parents=True, exist_ok=True)
    sm.narrative.path.write_text(text, encoding="utf-8")
    return sm


def test_others_hear_the_owners_bored_by_its_name(tmp_path):
    """Review r1: once a tone share is in the pair, members get a fresh read — on the axis that is "tired" now,
    while the owner hears the "bored" they asked for. They hear the same name."""
    sm = _model(tmp_path)
    lab, direction = M.resolve_label("bored")
    _set = datetime.datetime.now(datetime.timezone.utc)
    sm.state.set_owner_mood(lab, direction, 3)
    # the hold lasts to the END OF THE MOOD DAY (next local 04:00), not a fixed
    # span — "+10 h" was still inside it whenever the suite ran before ~18:00
    # (§4UR: failed at 07:59). One minute past the hold's own end.
    later = M.manner_ends_at(_set.isoformat()) + datetime.timedelta(minutes=1)
    assert sm.mood(now=later).label == "bored"
    _after, dp, why = sm.note_owner_sentiment("negative", 0.9, now=later, request_id="m1",
                                              emotions={"sadness": 0.9})
    assert why == "" and dp < 0
    mine = sm.mood(now=later, for_reply=True)
    other = sm.mood(now=later, for_reply=True, audience="others")
    assert mine.label == "bored" and other.label == "bored" and 0 < other.intensity <= mine.intensity


async def test_less_tired_on_the_axis_stays_on_the_axis(tmp_path, owner):
    """Review r1: "less tired" stepped onto the tired anchor (−0.75, −1) and added unpleasantness to a mood that
    had none (an automatic tired on the axis is new with §4UP)."""
    from ghost_agent.tools.introspect import tool_introspect
    sm = _model(tmp_path)
    sm.state.nudge(0.0, -0.5, "a very long turn", M.KIND_TURN)
    assert (sm.mood().label, sm.mood().intensity) == ("tired", 2)
    await tool_introspect(action="set_mood", self_model=sm, mood="tired", step=-1)
    v = sm.mood()
    assert (v.label, v.intensity) == ("tired", 1) and v.p == pytest.approx(0.0, abs=1e-9) and v.e < 0
    # a tired mood in the unpleasant quarter still steps down along its anchor, as before
    sm2 = _model(tmp_path / "q")
    sm2.state.nudge(-0.4, -0.5, "trouble and a long turn", M.KIND_TURN)
    await tool_introspect(action="set_mood", self_model=sm2, mood="tired", step=-1)
    v2 = sm2.mood()
    assert (v2.label, v2.intensity) == ("tired", 1) and v2.p < 0


def test_the_diary_date_is_a_mood_day(tmp_path):
    sm = _introspect_model(tmp_path, "x")
    import os
    t = datetime.datetime(2026, 11, 14, 1, 30, tzinfo=datetime.timezone.utc).timestamp()   # 03:30 Athens: the 13th
    os.utime(sm.narrative.path, (t, t))
    assert sm.narrative.written_on() == M.mood_day(datetime.datetime.fromtimestamp(t, datetime.timezone.utc)).isoformat()


def test_the_boot_line_lists_the_diary_only_when_it_is_on(monkeypatch):
    from ghost_agent.core import agent as A
    row = next(r for r in A.IDLE_SCHEDULER_JOBS if r[0] == "selfhood-narrative")
    ctx = SimpleNamespace(self_model=MagicMock(enabled=True))
    assert row[2](SimpleNamespace(), ctx) is False
    monkeypatch.setenv("GHOST_SELFHOOD_NARRATIVE", "1")
    assert A._selfhood_narrative_on() is True and row[2](SimpleNamespace(), ctx) is True


def test_the_last_entry_is_shown_dated_and_kept_out_of_the_summary(tmp_path, monkeypatch):
    from ghost_agent.tools import introspect as I
    sm = _introspect_model(tmp_path, "I keep reaching for the same tools.")
    shown = I._render_summary(sm)
    assert "reaching for the same tools" not in shown and "running first-person diary" not in shown
    monkeypatch.setenv("GHOST_SELFHOOD_NARRATIVE", "1")
    assert "reaching for the same tools" in I._render_summary(sm)
    assert sm.narrative.written_on() == datetime.datetime.now(datetime.timezone.utc).date().isoformat()
