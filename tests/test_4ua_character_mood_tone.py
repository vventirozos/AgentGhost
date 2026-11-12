"""§4UA: character, state (the mood pair) and expression (the tone lines) as one — four designed
couplings, each bounded.

(1) one ranking in the character section; (2) a reply written under a mood at 2+ or a manner never
feeds the daily look; (3) the character sets the mood's resting point, |rest| ≤ 0.12,
``GHOST_MIRROR_REST=0`` switches it off; (4) the face shows the rest faintly while the mood is neutral.
(§4UH-loop: written for the Mirror's sentence, whose look also asked for the rest and got a "lasting mood"
note; the Mirror is removed — the character's dials set the rest, ``character.rest_pair``.)"""
from __future__ import annotations

import asyncio
import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import loop as MI
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood.state import MOOD_HISTORY_FILENAME

UTC = datetime.timezone.utc
T0 = datetime.datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
H = M.HALF_LIFE_S


def _state(tmp_path, rest=None):
    st = SelfModel(root=tmp_path / "self").state
    if rest is not None:
        st.attach_rest(lambda: rest)
    return st


def _at(hours):
    return T0 + datetime.timedelta(hours=hours)


# ── (3) the resting point inside the mood engine ────────────────────────────

def test_rest_clamp_bounds_and_garbage():
    assert M.rest_clamp((0.5, -0.5)) == (M.REST_MAX, -M.REST_MAX)
    assert M.rest_clamp((0.05, -0.06)) == (0.05, -0.06)
    for bad in (None, "x", (1,), ("a", "b"), (float("nan"), 0.0), (float("inf"), 1)):
        got = M.rest_clamp(bad)
        assert all(abs(v) <= M.REST_MAX for v in got) and got in ((0.0, 0.0), (0.0, M.REST_MAX))


def test_rest_max_is_under_every_bound_a_level_is_held_to():
    """THE bound: rest alone never makes a mood anyone hears, whatever level was held."""
    assert M.REST_MAX < M.INTENSITY_THRESHOLDS[0] - M.INTENSITY_MARGIN
    for held in (0, 1, 2, 3):
        for rp in (-M.REST_MAX, 0.0, M.REST_MAX):
            for re_ in (-M.REST_MAX, 0.0, M.REST_MAX):
                assert M.read_off(rp, re_, held_label="content", held_intensity=held)[1] == 0


def _views(st, now):
    return {"owner": st.mood(now=now), "reply": st.mood(now=now, for_reply=True),
            "others": st.mood(now=now, audience="others"), "public": st.mood(now=now, audience="public")}


def _script(st):
    """One day of the mood: events, a sentiment (the owner's share), a set mood, a step, a reset."""
    st.nudge(0.10, 0.05, "a clean turn", M.KIND_TURN, now=_at(0))
    st.nudge(0.12, 0.02, "the owner sounded pleased", M.KIND_SENTIMENT, now=_at(0.5))
    yield "events"
    st.nudge(-0.30, 0.20, "two tools failed", M.KIND_TURN, now=_at(1))
    yield "trouble"
    st.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 3, now=_at(2), hold=False)
    yield "owner set"
    st.settle(now=_at(5))
    yield "decayed"
    st.reset_mood(now=_at(6))
    yield "reset"


@pytest.mark.parametrize("hours", [0.0, 0.7, 3.0, 9.0])
def test_with_rest_zero_every_view_is_what_it_was_without_one(tmp_path, hours):
    """Everything §4TC/§4TX guaranteed holds with rest (0, 0) — the same views, step by step."""
    a, b = _state(tmp_path / "a"), _state(tmp_path / "b", rest=(0.0, 0.0))
    for _ in zip(_script(a), _script(b)):
        now = _at(6 + hours)
        assert _views(a, now) == _views(b, now)
    assert a.path.read_bytes() == b.path.read_bytes()


REST = (0.12, -0.06)


@pytest.mark.parametrize("rest", [REST, (-0.12, 0.12), (0.06, 0.0)])
def test_every_reader_sees_rest_plus_the_stored_deviation(tmp_path, rest):
    """R5: one input, one story — every view reports rest + deviation and the label read off the deviation."""
    st = _state(tmp_path, rest=rest)
    for step in _script(st):
        for hours in (0.0, 1.5, 7.0):
            now = _at(6 + hours) if step == "reset" else _at({"events": 0.6, "trouble": 1.1,
                                                                "owner set": 2.1, "decayed": 5.0}[step] + hours)
            stored = st.state.mood
            age = st._age(stored, now)
            dev = (M.decay(stored.p, age), M.decay(stored.e, age)) if age is not None else (0.0, 0.0)
            v = st.mood(now=now)
            assert (v.p, v.e) == (round(M.clamp(dev[0] + rest[0]), 4), round(M.clamp(dev[1] + rest[1]), 4)), \
                (step, hours)
            label, n, _ = M.read_off(dev[0], dev[1], held_label=stored.label,
                                     held_intensity=stored.intensity, novelty=stored.novelty)
            if not (v.label == M.BORED and n == 0):
                assert (v.label, v.intensity) == (label, n), (step, hours)


def test_a_mood_decays_to_the_rest_not_to_zero_and_reads_neutral_there(tmp_path):
    st = _state(tmp_path, rest=REST)
    st.nudge(0.6, 0.0, "praise", M.KIND_PRAISE, now=T0)
    late = st.mood(now=_at(48))
    assert late.neutral and (late.p, late.e) == pytest.approx(REST, abs=1e-4)


def test_an_owner_set_mood_is_exactly_the_label_and_level_asked_for(tmp_path):
    """Every label, every level, every corner of the rest: the label and level the owner named, read
    the same by every listener and after hours of hold; the numbers are anchor + rest, never past ±1."""
    corners = [(a, b) for a in (-0.12, 0.0, 0.12) for b in (-0.12, 0.0, 0.12)]
    for rest in corners:
        st = _state(tmp_path / str(rest), rest=rest)
        for label in M.ANCHOR_DIRECTION:
            for n in (1, 2, 3):
                st.set_owner_mood(label, M.ANCHOR_DIRECTION[label], n, now=T0)
                for now in (T0, _at(3)):
                    for v in (st.mood(now=now), st.mood(now=now, audience="others")):
                        assert (v.label, v.intensity) == (label, n), (rest, label, n)
                v = st.mood(now=T0)
                want = M.anchor_for(M.ANCHOR_DIRECTION[label], n)
                assert (v.p, v.e) == pytest.approx((M.clamp(want[0] + rest[0]), M.clamp(want[1] + rest[1])))
                assert v.held


def test_neutral_from_the_owner_rests_at_the_character(tmp_path):
    st = _state(tmp_path, rest=REST)
    st.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 3, now=T0)
    _b, after = st.reset_mood(now=T0)
    assert after.neutral and (after.p, after.e) == pytest.approx(REST)


def test_others_never_hear_more_than_the_owner_with_a_rest(tmp_path):
    """§4TX's promise survives the rest: others' level ≤ the owner's, on a grid of shares and rests."""
    for rest in ((0.12, 0.0), (-0.12, 0.0), (0.0, 0.12), (0.12, -0.12)):
        for share in (0.15, 0.3, -0.15, -0.3):
            st = _state(tmp_path / f"{rest}{share}", rest=rest)
            st.nudge(0.1, 0.05, "a clean turn", M.KIND_TURN, now=T0)
            st.nudge(share, share / 3, "the owner's tone", M.KIND_SENTIMENT, now=T0)
            for hours in (0.0, 1.0, 3.0):
                own, oth = st.mood(now=_at(hours)), st.mood(now=_at(hours), audience="others")
                assert oth.intensity <= own.intensity, (rest, share, hours)
                assert oth.neutral or oth.label == own.label or _same_side(oth, own), (rest, share, hours)


def _same_side(a, b):
    """Never the OPPOSITE pleasantness or energy of the owner's mood (review r1 MAJOR 1)."""
    da, db = M.ANCHOR_DIRECTION[a.label], M.ANCHOR_DIRECTION[b.label]
    return all(x * y >= 0 for x, y in zip(da, db))


def test_review_r1_others_never_hear_the_opposite_pleasantness(tmp_path):
    st = _state(tmp_path, rest=(-0.12, 0.0))
    st.nudge(0.30, 0.0, "the owner sounded delighted", M.KIND_SENTIMENT, now=T0)
    st.nudge(0.0, 0.25, "research", M.KIND_TURN, now=T0)
    own, oth = st.mood(now=T0), st.mood(now=T0, audience="others")
    assert own.label == "cheerful" and oth.label in ("restless", "neutral"), (own, oth)
    assert st.mood(now=T0, audience="public").label in ("restless", "neutral")


@pytest.mark.parametrize("rest", [(0.0, 0.12), (0.0, -0.12), (0.12, 0.12), (-0.12, -0.12)])
@pytest.mark.parametrize("dev", [(-0.5, 0.0), (0.0, 0.6), (0.5, -0.3), (-0.3, -0.7)])
def test_review_r1_decay_never_changes_the_label(tmp_path, rest, dev):
    """A mood fades in strength only: never into another mood, read or settled, for any rest."""
    st = _state(tmp_path, rest=rest)
    st.nudge(*dev, "an event", M.KIND_TURN, now=T0)
    first = st.mood(now=T0).label
    seen = set()
    for minutes in range(0, 12 * 60, 10):
        v = st.mood(now=T0 + datetime.timedelta(minutes=minutes))
        seen.add(v.label if v.intensity else M.NEUTRAL)
    assert seen <= {first, M.NEUTRAL}, (first, seen)
    st.settle(now=_at(2.5))
    st.settle(now=_at(12))
    assert {r["label"] for r in st.mood_history(limit=20)} <= {first, M.NEUTRAL}


def test_review_r1_a_rest_moved_overnight_raises_no_mood(tmp_path):
    rest = [0.06, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(rest))
    st.nudge(0.36, 0.0, "praise", M.KIND_PRAISE, now=T0)
    assert (st.mood(now=T0).label, st.mood(now=T0).intensity) == ("content", 1)
    rows = len(st.mood_history(limit=50))
    rest[0] = 0.12                                       # the next look moved the character
    v = st.settle(now=T0)
    assert (v.label, v.intensity) == ("content", 1) and len(st.mood_history(limit=50)) == rows
    # … nor makes one from a deviation that was nothing
    st.reset_mood(now=_at(1))
    rest[0] = 0.06
    st.nudge(0.10, 0.0, "a small thing", M.KIND_TURN, now=_at(1))
    assert st.mood(now=_at(1)).neutral                  # 0.10 + 0.06 = 0.16: nothing
    rest[0] = 0.12
    assert st.mood(now=_at(1)).neutral                  # 0.10 + 0.12 = 0.22 would be "content, 1": no rise on a read


def test_review_r1_the_pair_as_read_stays_in_range(tmp_path):
    st = _state(tmp_path, rest=(0.12, -0.12))
    for _ in range(6):
        st.nudge(0.5, -0.5, "a lot", M.KIND_TURN, now=T0)
    v = st.mood(now=T0)
    assert (v.p, v.e) == (1.0, -1.0) and st.mood_history(limit=1)[-1]["p"] <= 1.0


def test_a_mood_that_wore_off_unseen_is_dated_when_it_did_with_a_rest(tmp_path):
    st = _state(tmp_path, rest=(0.12, 0.0))
    st.nudge(0.5, 0.0, "praise", M.KIND_PRAISE, now=T0)        # the deviation 0.5 → content 2
    assert st.mood(now=T0).intensity == 2
    v = st.settle(now=_at(10))
    assert v.neutral
    want = T0 + datetime.timedelta(seconds=M.decay_step_s(0.5, 0))      # the rest does not move the dating
    rows = st.mood_history(limit=5)
    assert rows[-1]["at"] == want.replace(microsecond=0, tzinfo=None).isoformat() + "Z"
    # the journal row carries the pair as read
    assert rows[-1]["p"] == pytest.approx(round(0.12 + M.decay(0.5, 10 * 3600), 3), abs=1e-3)


def test_a_rest_that_cannot_be_read_is_no_rest(tmp_path):
    def boom():
        raise OSError("gone")
    st = _state(tmp_path)
    st.attach_rest(boom)
    assert st.rest() == (0.0, 0.0) and st.mood(now=T0).neutral
    st.attach_rest(lambda: ("x", None))
    assert st.rest() == (0.0, 0.0)
    st.attach_rest(lambda: (5.0, -5.0))
    assert st.rest() == (M.REST_MAX, -M.REST_MAX)


# ── (2) the daily look: what it is shown ─────────────────────────────────────

NOW = datetime.datetime(2026, 11, 10, 3, 0, 0)


def _turn(h, tone=None, text=None):
    now_utc = NOW.astimezone(UTC)
    t = {"final_response": text or f"Reply {h}h: " + "a real answer, long enough to be picked. " * 3,
         "timestamp": (now_utc - datetime.timedelta(hours=h)).isoformat()}
    if tone is not None:
        t["extra"] = {"mood_tone": tone}
    return t


def _tone(label, n, manner="", mn=0):
    """A turn record's ``extra.mood_tone`` as the agent writes it (§4UD W2: the values beside the sentences)."""
    return {"mood": (f"{label}, {n} of 3" if n else "neutral"), "manner": (f"{manner}, {mn} of 3" if manner else ""),
            "label": label, "intensity": n, "manner_name": manner, "manner_intensity": mn}


@pytest.mark.parametrize("tone,kept", [
    (None, True),
    (_tone("content", 1), True),
    (_tone("neutral", 0), True),
    (_tone("irritable", 2), False),
    (_tone("cheerful", 3), False),
    (_tone("neutral", 0, "sarcastic", 1), False),
    (_tone("content", 1, "terse", 2), False),
    ("garbage", True),
    # the VALUES decide, never the sentence (it was parsed back with a regex): a record whose sentence
    # says 3 and whose value says 1 was written under a faint mood
    ({"mood": "cheerful, 3 of 3", "manner": "", "label": "cheerful", "intensity": 1, "manner_name": ""}, True),
    ({"mood": "neutral", "manner": "", "label": "down", "intensity": 2, "manner_name": ""}, False),
    ({"mood": "neutral", "manner": "", "label": "neutral", "intensity": True, "manner_name": ""}, True),
    # a record from before §4UD has only the sentences: they are not read at all (review r2: counting any
    # mood in them as strong could skip the look for two days after a deploy)
    ({"mood": "content, 1 of 3", "manner": ""}, True),
    ({"mood": "irritable, 3 of 3", "manner": ""}, True),
    ({"mood": "neutral", "manner": "sarcastic, 1 of 3"}, True),
    ({"mood": 7}, True),
    ({"mood": "neutral", "manner": "", "manner_name": 7, "intensity": "3"}, True),
])
def test_a_reply_written_under_a_strong_mood_or_a_manner_is_never_shown(tmp_path, monkeypatch, tone, kept):
    import ghost_agent.core.failure_census as FC
    turns = [_turn(1, tone), _turn(2)]
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, days: (turns, []))
    got = MI.pick_replies(tmp_path, now=NOW)
    assert any(g["text"].startswith("Reply 1h") for g in got) is kept
    assert any(g["text"].startswith("Reply 2h") for g in got)


def test_a_turn_record_with_a_non_dict_extra_is_no_mood():
    assert MI.written_under_strong_mood({"extra": "x"}) is False
    assert MI.written_under_strong_mood({"extra": None}) is False
    assert MI.written_under_strong_mood({}) is False


def test_the_real_tone_record_is_what_the_filter_reads(tmp_path):
    """Review r1 n5: the record built by core/agent.py's own tone block, not a hand-written dict."""
    from ghost_agent.core import agent as A
    from ghost_agent.utils.logging import request_id_context
    for level, manner, strong in ((1, "", False), (2, "", True), (3, "", True), (1, "terse", True)):
        sm = SelfModel(root=tmp_path / f"s{level}{manner}")
        sm.state.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], level)
        if manner:
            sm.state.set_manner(manner, 1)
        ctx = SimpleNamespace(self_model=sm, turn_origin_label=None, owner_memory_isolated=False,
                              _replay_forbidden_tools=None)
        rid = f"web-4ua-{level}{manner}"
        tok = request_id_context.set(rid)
        try:
            assert A._mood_tone_block(ctx)
        finally:
            request_id_context.reset(tok)
        rec = A.mood_tone_heard(rid)
        assert MI.written_under_strong_mood({"extra": {"mood_tone": rec}}) is strong, rec


# ── (3) the resting point the character sets (§4UH-e2e; the Mirror's look no longer asks) ─────────

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv(MI.REST_ENV, raising=False)
    monkeypatch.delenv(C.ENV, raising=False)
    C._MEMO.clear()
    (tmp_path / "system" / "memory").mkdir(parents=True)
    return tmp_path


def test_the_switch_turns_the_rest_off_everywhere(home, monkeypatch):
    C.set_dial(home, "humour", 2)                                   # warmth -1, humour +2: a little sunny, brisk
    assert C.rest_pair(home) == pytest.approx((0.06, 0.06))
    assert "my mood rests at pleasantness a little sunny" in MI.banner(home)
    monkeypatch.setenv(MI.REST_ENV, "0")
    assert C.rest_pair(home) == (0.0, 0.0)
    assert "my mood rests" not in MI.banner(home)
    monkeypatch.setenv(MI.REST_ENV, "1")
    monkeypatch.setenv(C.ENV, "0")                                  # §4UH-loop: the character off is no rest
    assert C.rest_pair(home) == (0.0, 0.0)


def test_the_rest_stays_in_range_at_every_dial(home):
    for w in range(-2, 3):
        for h in range(-2, 3):
            a, b = C.rest_units({"warmth": w, "humour": h, "directness": 0})
            assert all(abs(v) * MI.REST_UNIT <= M.REST_MAX + 1e-9 for v in (a, b))


def test_the_live_wiring_the_mood_reads_the_characters_rest(home):
    """main.py's lambda, through the real layout: memory_dir = $HOME/system/memory."""
    C.set_dial(home, "humour", 2)
    C.set_dial(home, "warmth", 0)                                   # s = 2 → p 1, e 1
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"))
    sm.attach_rest(lambda: C.rest_pair(MI.home_from_memory_dir(getattr(ctx, "memory_dir", None))))
    v = sm.state.mood(now=T0)
    assert v.neutral and (v.p, v.e) == pytest.approx((0.06, 0.06))


def test_main_attaches_the_rest_from_the_character():
    """The one production call site, read off the syntax tree (a call in a comment is not a call)."""
    import ast
    src = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "main.py"
    calls = [n for n in ast.walk(ast.parse(src.read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "attach_rest"]
    assert len(calls) == 1
    inner = [c for c in ast.walk(calls[0]) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "rest_pair"]
    assert len(inner) == 1


# ── (1) one ranking ──────────────────────────────────────────────────────────

def test_the_character_section_states_the_ranking_and_yields_to_mood_and_manner(tmp_path):
    rule = C.RULE
    assert ("Precedence: the owner's rules, then correctness, then the manner, then the mood, then your "
            "character.") in rule
    assert "when a MOOD or MANNER line is present, that line comes first for this reply" in rule
    # the mood's own ranking is unchanged and agrees: manner before mood
    assert "Precedence: the owner's rules, then correctness, then the manner, then the mood." in \
        M.tone_line("irritable", 2)


# ── (4) the face and the handheld ────────────────────────────────────────────

def test_health_carries_the_rest_label(tmp_path):
    # §4UD r2: routes._rest_label is gone — the rule is the engine's `mood.rest_label`, read through health
    from ghost_agent.api import routes

    def rest_of(st):
        return routes._health_mood(SimpleNamespace(self_model=SimpleNamespace(state=st)))["rest"]
    assert rest_of(_state(tmp_path, rest=(0.12, -0.06))) == "calm"
    assert rest_of(_state(tmp_path / "z")) == "neutral"
    assert rest_of(SimpleNamespace()) == "neutral"
    # review r1 m2: an unhurried character is not bored (boredom is a quiet, not a temperament)
    assert rest_of(_state(tmp_path / "b", rest=(0.0, -0.12))) == "calm"
    assert rest_of(_state(tmp_path / "d", rest=(-0.12, 0.0))) == "down"


def test_introspect_names_the_resting_point_for_the_owner_only(tmp_path, monkeypatch):
    from ghost_agent.tools import introspect as I
    st = _state(tmp_path, rest=(0.12, -0.06))
    sm = SimpleNamespace(state=st)
    monkeypatch.setattr(I, "_reply_is_public", lambda: False)
    text = I._render_mood(sm)
    assert I.REST_TEXT.format(p=0.12, e=-0.06) in text
    monkeypatch.setattr(I, "_reply_is_public", lambda: True)
    assert "Resting point" not in I._render_mood(sm)
    monkeypatch.setattr(I, "_reply_is_public", lambda: False)
    assert "Resting point" not in I._render_mood(SimpleNamespace(state=_state(tmp_path / "z")))


class _Resp:
    def __init__(self, body, code=200):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


class _Client:
    def __init__(self, body):
        self.body = body

    async def get(self, url, headers=None):
        return _Resp(self.body)


@pytest.mark.parametrize("mood,want", [
    ({"label": "neutral", "intensity": 0, "rest": "calm"}, ("calm", "rest")),
    ({"label": "irritable", "intensity": 2, "rest": "calm"}, ("irritable", 2)),
    ({"label": "neutral", "intensity": 0, "rest": "neutral"}, ("neutral", 0)),
    ({"label": "neutral", "intensity": 0}, ("neutral", 0)),
    ({"label": "neutral", "intensity": 0, "rest": "Calm!"}, ("neutral", 0)),
    ({"label": "neutral", "intensity": 0, "rest": 3}, ("neutral", 0)),
])
def test_the_handheld_shows_the_rest_while_neutral(mood, want):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interface" / "externals" / "clockwork_ghost"))
    try:
        import agentapi
    finally:
        sys.path.pop(0)
    assert asyncio.run(agentapi.fetch_mood(_Client({"mood": mood}), "http://x", "k")) == want


# ── review r2 ───────────────────────────────────────────────────────────────

def test_review_r2_an_event_the_other_way_does_not_finish_a_rest_rise(tmp_path):
    """r2 M1: the look moved the rest up a unit; the next event — the owner
    sounding a little down — read the level the rest had made."""
    rest = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(rest))
    st.nudge(0.43, 0.0, "praise", M.KIND_PRAISE, now=T0)
    assert (st.mood(now=T0).label, st.mood(now=T0).intensity) == ("content", 1)
    rest[0] = 0.06
    assert st.mood(now=T0).intensity == 1
    st.nudge(-0.02, -0.005, "the owner sounded a little down", M.KIND_SENTIMENT, now=T0)
    v = st.mood(now=T0)
    assert (v.label, v.intensity) == ("content", 1), v
    # an event that itself carries the strength over the next bound still raises it
    st.nudge(0.35, 0.0, "a task finished", M.KIND_TURN, now=T0)
    assert st.mood(now=T0).intensity == 3


@pytest.mark.parametrize("rest", [(0.12, 0.0), (0.0, 0.0)])
def test_review_r2_a_changed_thumb_leaves_nothing_behind_at_the_limit(tmp_path, rest):
    """r2 M2: the ledger booked the VIEWS' move (rest + deviation, clamped at
    1) — a thumb near the limit was booked as no move, and toggling left
    +0.10 in the pair."""
    sm = SelfModel(root=tmp_path / "self")
    sm.attach_rest(lambda: rest)
    sm.state.nudge(0.90, 0.0, "seed", M.KIND_TURN, now=T0)
    for k in ("thumb_up", "thumb_down", "thumb_up", "thumb_down"):
        sm.note_owner_reaction(k, now=T0, reply_id="r1")
    assert sm.state._state.mood.p == pytest.approx(0.60, abs=1e-6)
    sm.note_owner_reaction("thumb_up", now=T0, reply_id="r1")
    assert sm.state._state.mood.p == pytest.approx(1.0, abs=1e-6)        # the up replaces the down
    assert sm._applied["r1"][0] == pytest.approx(0.10, abs=1e-4)          # booked what the store moved


def test_review_r2_nudge_reports_the_stored_move_not_the_views(tmp_path):
    st = _state(tmp_path, rest=(0.12, 0.0))
    st.nudge(0.95, 0.0, "seed", M.KIND_TURN, now=T0)
    before, after = st.nudge(0.03, 0.0, "x", M.KIND_TURN, now=T0)
    assert before.p == after.p == 1.0                 # the views saturate
    assert after.moved == pytest.approx((0.03, 0.0))  # the store moved


def test_review_r2_at_rest_zero_the_numbers_are_written_as_before(tmp_path):
    """r2 m2: clamp(-0.0 + 0.0) wrote 0.0 where the journal had -0.0."""
    st = _state(tmp_path, rest=(0.0, 0.0))
    st.nudge(0.3, -0.0, "x", M.KIND_TURN, now=T0)
    raw = json.loads(st.path.read_text())
    v = st.mood(now=T0)
    assert math.copysign(1.0, v.e) == math.copysign(1.0, raw["mood"]["e"])


def test_review_r2_others_are_capped_at_the_owners_level(tmp_path):
    """Pin for a battery survivor: others' level is capped at the owner's even
    where the toward-zero share would read stronger beside the rest."""
    st = _state(tmp_path, rest=(0.12, 0.0))
    st.nudge(0.10, 0.0, "the owner sounded pleased", M.KIND_SENTIMENT, now=T0)
    st.nudge(0.10, 0.0, "a clean turn", M.KIND_TURN, now=T0)
    st.set_owner_mood("content", M.ANCHOR_DIRECTION["content"], 1, now=_at(0.1), hold=False)
    for h in (0.1, 0.5, 1.5, 3.0):
        own, oth = st.mood(now=_at(h)), st.mood(now=_at(h), audience="others")
        assert oth.intensity <= own.intensity, (h, own, oth)


@pytest.mark.parametrize("rest", [(0.12, 0.12), (-0.12, -0.12)])
def test_review_r2_shown_numbers_stay_in_range(tmp_path, rest):
    """Pin for a battery survivor: rest + deviation is clamped in every view."""
    st = _state(tmp_path, rest=rest)
    sign = 1 if rest[0] > 0 else -1
    before, after = st.nudge(sign * 2.0, sign * 2.0, "x", M.KIND_TURN, now=T0)
    for v in (after, st.mood(now=T0), st.mood(now=T0, audience="others")):
        assert -1.0 <= v.p <= 1.0 and -1.0 <= v.e <= 1.0, v
    for r in st.mood_history(limit=5):
        assert -1.0 <= r["p"] <= 1.0 and -1.0 <= r["e"] <= 1.0


# ── review r3 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("r,want", [(0.0, ("content", 1)), (0.06, ("content", 1)), (0.12, ("content", 1))])
def test_review_r3_praise_after_a_rest_rise_is_heard(tmp_path, r, want):
    """r3: after the rest moved up, a stored neutral whose unheld level was 1
    swallowed a praise (neutral for hours): a sunny rest made praise weaker
    than no rest at all. The event made the mood stronger, so it may rise."""
    rest = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(rest))
    st.nudge(0.15, 0.0, "a clean turn", M.KIND_TURN, now=T0)
    assert st.mood(now=T0).intensity == 0
    rest[0] = r
    st.nudge(0.20, 0.0, "praise", M.KIND_PRAISE, now=T0)
    v = st.mood(now=T0)
    assert (v.label, v.intensity) == want, v
    assert st.mood(now=_at(1)).intensity >= 1


def test_review_r3_a_near_tie_is_read_as_it_is_stored(tmp_path):
    """r3: a near-tie of the axes became an exact tie on the 4-dp write, the
    tilt moved axis and the level dropped the moment it was written (kept
    after the tilt was removed: the read and the stored pair agree)."""
    st = _state(tmp_path, rest=(-0.12, 0.12))
    _, after = st.nudge(0.349996, 0.35, "x", M.KIND_TURN, now=T0)
    again = st.mood(now=T0)
    assert (after.label, after.intensity) == (again.label, again.intensity)
    assert [(r["label"], r["intensity"]) for r in st.mood_history(limit=5)] == [(again.label, again.intensity)]


# ── review r4 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed,rest,event", [
    ((0.40, 0.36), (0.06, 0.12), (-0.20, 0.0)),       # a correction on the stronger axis
    ((0.34, -0.33), (0.12, -0.12), (-0.08, -0.02)),   # "the owner sounded a little down"
    ((0.40, 0.38), (0.0, 0.12), (-0.20, 0.0)),        # no rest move at all
])
def test_review_r4_an_event_that_swaps_the_stronger_axis_raises_nothing(tmp_path, seed, rest, event):
    """r4 (kept as a guard after the tilt was removed): the tilt sat on the stronger axis; an event that made the other
    axis the stronger one raised the strength although no axis grew."""
    r = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(r))
    st.nudge(*seed, "seed", M.KIND_TURN, now=T0)
    r[:] = rest
    n0 = st.mood(now=T0).intensity
    st.nudge(*event, "the owner corrected the reply", M.KIND_CORRECTION, now=T0)
    assert st.mood(now=T0).intensity <= n0


def test_review_r4_an_event_on_the_weak_axis_that_adds_nothing_does_not_rise(tmp_path):
    """Pin for the strict test: equal strength is no rise."""
    r = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(r))
    st.nudge(0.36, 0.0, "seed", M.KIND_TURN, now=T0)
    r[:] = (0.12, 0.12)
    n0 = st.mood(now=T0).intensity
    st.nudge(*M.EV_LONG_CLEAN_TURN, "a long clean turn", M.KIND_TURN, now=T0)
    assert st.mood(now=T0).intensity == n0


def test_review_r4_a_shrinking_event_never_raises_the_level_property(tmp_path):
    """The invariant behind r2–r4, swept: beside any rest, an event that moves
    no axis outward (each |deviation| no larger, same sign or zero) never
    raises the level — after a rest change, at a hysteresis hold, on ties."""
    import random
    rnd = random.Random(4)
    grid = [-0.12, -0.06, 0.0, 0.06, 0.12]
    bad = []
    for i in range(1500):
        r = [0.0, 0.0]
        st = SelfModel(root=tmp_path / f"s{i}").state
        st.attach_rest(lambda: tuple(r))
        seed = (round(rnd.uniform(-0.9, 0.9), 2), round(rnd.uniform(-0.9, 0.9), 2))
        if rnd.random() < 0.2:
            seed = (seed[0], math.copysign(abs(seed[0]), seed[1]))          # a tie
        st.nudge(*seed, "seed", M.KIND_TURN, now=T0)
        r[:] = (rnd.choice(grid), rnd.choice(grid))
        n0 = st.mood(now=T0).intensity
        base = st._state.mood
        if base is None:
            continue
        ev = []
        for d in (base.p, base.e):                    # shrink each axis by 0..100 %
            ev.append(-d * rnd.choice([0.0, 0.0, 0.05, 0.3, 0.7, 1.0]))
        if ev == [0.0, 0.0]:
            continue
        st.nudge(*ev, "x", M.KIND_CORRECTION, now=T0)
        if st.mood(now=T0).intensity > n0:
            bad.append((seed, tuple(r), ev))
        if len(bad) > 3:
            break
    assert not bad, bad


def test_review_r4_an_event_that_builds_the_weak_axis_past_the_strong_one_is_heard(tmp_path):
    """Battery r4: the rise is measured on the axis the mood had. Building the
    weak axis past it is a real event, heard as it would be with no rest."""
    r = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(r))
    st.nudge(0.18, 0.0, "seed", M.KIND_TURN, now=T0)
    r[:] = (0.12, -0.12)
    assert st.mood(now=T0).intensity == 0
    st.nudge(0.0, 0.36, "research", M.KIND_TURN, now=T0)
    assert st.mood(now=T0).intensity == 1


# ── review r5 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed,rest,event", [
    ((-0.19, 0.13), (0.12, 0.06), (0.08, 0.02)),      # the owner sounded pleased
    ((0.19, -0.17), (-0.12, -0.12), (-0.08, -0.02)),  # the owner sounded a little down
    ((0.09, 0.11), (-0.12, -0.06), (-0.20, 0.0)),     # a correction
])
def test_review_r5_a_small_push_past_the_other_axis_does_not_carry_the_whole_rest(tmp_path, seed, rest, event):
    """r5 (kept as a guard after the tilt was removed): the tilt moved with the stronger axis, so a 0.02 push that made the
    weaker axis the stronger one gave a whole level, often against the event."""
    r = [0.0, 0.0]
    st = _state(tmp_path)
    st.attach_rest(lambda: tuple(r))
    st.nudge(*seed, "seed", M.KIND_TURN, now=T0)
    r[:] = rest
    assert st.mood(now=T0).intensity == 0
    st.nudge(*event, "x", M.KIND_SENTIMENT, now=T0)
    assert st.mood(now=T0).intensity == 0, st.mood(now=T0)


# ── operator 11-10: the rest SHOWS, it never tilts ──────────────────────────

def test_the_heard_mood_is_the_same_at_every_rest_property(tmp_path):
    """After five review rounds of a rest that tilted the level (each MAJOR
    inside the previous fix), the operator chose: the rest moves the numbers,
    the face and introspect; which mood is heard, at what level, since when
    and why is exactly what it is at rest (0, 0) — for the owner, a reply,
    others and the public, through events, sentiment shares, owner sets,
    holds, decay and rest changes. The numbers are rest + deviation."""
    import random
    rnd = random.Random(11)
    grid = [-0.12, -0.06, 0.0, 0.06, 0.12]
    kinds = [M.KIND_TURN, M.KIND_SENTIMENT, M.KIND_PRAISE, M.KIND_CORRECTION]
    for run in range(60):
        r = [0.0, 0.0]
        a = SelfModel(root=tmp_path / f"a{run}").state
        b = SelfModel(root=tmp_path / f"b{run}").state
        b.attach_rest(lambda: tuple(r))
        t = T0
        for step in range(30):
            t = t + datetime.timedelta(minutes=rnd.choice([0, 5, 30, 90, 240]))
            x = rnd.random()
            if x < 0.15:
                r[:] = (rnd.choice(grid), rnd.choice(grid))       # a look moved the rest
            elif x < 0.75:
                ev = (round(rnd.uniform(-0.4, 0.4), 3), round(rnd.uniform(-0.4, 0.4), 3), "c",
                      rnd.choice(kinds))
                a.nudge(*ev, now=t, novelty=step % 7 == 0)
                b.nudge(*ev, now=t, novelty=step % 7 == 0)
            elif x < 0.82:
                lab = rnd.choice(sorted(M.ANCHOR_DIRECTION))
                lvl, hold = rnd.choice([1, 2, 3]), rnd.random() < 0.5
                a.set_owner_mood(lab, M.ANCHOR_DIRECTION[lab], lvl, now=t, hold=hold)
                b.set_owner_mood(lab, M.ANCHOR_DIRECTION[lab], lvl, now=t, hold=hold)
            elif x < 0.88:
                a.settle(now=t)
                b.settle(now=t)
            elif x < 0.92:
                a.reset_mood(now=t)
                b.reset_mood(now=t)
            elif x < 0.95:
                m = rnd.choice(["terse", ""])
                a.set_manner(m, 2 if m else 0, now=t)
                b.set_manner(m, 2 if m else 0, now=t)
            quiet = rnd.choice([None, None, 20000.0, 50000.0])     # review r7: the boredom path too
            for kw in ({}, {"for_reply": True}, {"audience": "others"}, {"audience": "public"},
                       {"quiet_s": quiet}, {"quiet_s": quiet, "audience": "others"}):
                va, vb = a.mood(now=t, **kw), b.mood(now=t, **kw)
                assert (va.label, va.intensity, va.set_at, va.cause, va.source, va.held_until, va.recent) == \
                    (vb.label, vb.intensity, vb.set_at, vb.cause, vb.source, vb.held_until, vb.recent), \
                    (run, step, kw, va, vb, tuple(r))
                assert vb.p == pytest.approx(M.clamp(va.p + r[0]), abs=2e-4), (run, step, kw)
                assert vb.e == pytest.approx(M.clamp(va.e + r[1]), abs=2e-4), (run, step, kw)
        ha = [(x["label"], x["intensity"], x["at"], x["kind"]) for x in a.mood_history(limit=200)]
        hb = [(x["label"], x["intensity"], x["at"], x["kind"]) for x in b.mood_history(limit=200)]
        assert ha == hb, run


def test_the_tilt_is_gone_from_the_engine():
    """No read-time tilt survives to be wired back by accident."""
    for name in ("rest_strength", "rest_decay_step_s", "read_off_rest", "_tilt", "_axis_strength"):
        assert not hasattr(M, name), name



def test_battery_the_journal_file_itself_holds_numbers_in_range(tmp_path):
    """Battery survivor: the reader clamps on load, so the WRITE's clamp is
    pinned on the raw file (another reader of the file gets 1.12 otherwise)."""
    st = _state(tmp_path, rest=(0.12, 0.12))
    st.nudge(2.0, 2.0, "x", M.KIND_TURN, now=T0)
    for line in st.mood_history_path.read_text().splitlines():
        row = json.loads(line)
        assert -1.0 <= row["p"] <= 1.0 and -1.0 <= row["e"] <= 1.0, row


def test_battery_at_rest_zero_a_negative_zero_is_kept():
    """Battery survivor: a deviation that rounds to -0.0 is shown as -0.0 at
    rest 0, exactly as before §4UA (``clamp(-0.0 + 0.0)`` is +0.0)."""
    from ghost_agent.selfhood.state import _shown
    assert math.copysign(1.0, _shown(-0.00001, 0.0)) == -1.0
    assert math.copysign(1.0, _shown(-0.0, 0.0)) == -1.0      # clamp(-0.0 + 0.0) is +0.0
    assert _shown(-0.00001, 0.12) == pytest.approx(0.12)


def test_battery_the_health_endpoint_carries_the_rest_region(tmp_path):
    """Battery survivor: ``_rest_label`` was pinned, its use in /api/health was not."""
    from unittest.mock import MagicMock
    from ghost_agent.api import routes
    st = _state(tmp_path, rest=(0.12, -0.06))
    agent = SimpleNamespace(context=SimpleNamespace(self_model=SimpleNamespace(state=st),
                                                     llm_client=None, scheduler=None, memory_system=None))
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=agent, biological_task=None, boot_monotonic=None,
                                                    resolved_config={}))
    body = json.loads(asyncio.new_event_loop().run_until_complete(routes.api_health(req)).body)
    assert body["mood"]["rest"] == "calm" and body["mood"]["intensity"] == 0
