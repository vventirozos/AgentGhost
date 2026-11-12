"""§4UH: the character moves from the agent's own days and from its own look at its replies; it has
tastes; and it leans on how hard the events of a day land."""
from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import mirror as MIRROR
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import selfview as V

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def _home(tmp_path) -> Path:
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "selfhood").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    C._MEMO.clear()


class _LLM:
    def __init__(self, answer="0 1 -1", error=None):
        self.answer, self.error, self.asked = answer, error, []

    async def chat_completion(self, payload, **kw):
        self.asked.append((payload, kw))
        if self.error:
            raise self.error
        return {"choices": [{"message": {"content": self.answer}}]}


def _replies(monkeypatch, n=4):
    picked = [{"text": f"reply number {i} " + "x" * 100, "at": f"2026-11-12T0{i}:00:00Z"} for i in range(n)]
    monkeypatch.setattr(MIRROR, "pick_replies", lambda home, now=None, rng=None, after="": list(picked))
    return picked


# ── the look ────────────────────────────────────────────────────────────────

def test_the_look_asks_for_three_numbers_and_never_shows_the_dials(tmp_path, monkeypatch):
    home = _home(tmp_path)
    C.set_dial(home, "humour", 2, now=NOW)
    _replies(monkeypatch)
    llm = _LLM("0 1 -1")
    note = asyncio.run(V.look(home, llm, now=NOW))
    assert note == "looked at 4 replies: warmth +0 (dial -1), humour +1 (dial +2), directness -1 (dial +0)"
    prompt = llm.asked[0][0]["messages"][1]["content"]
    assert "reply number 0" in prompt and "warmth: -2 dry and unsentimental" in prompt and "never to be kind" in prompt
    assert C.sentence(C.load(home)["dials"]) not in prompt and "dial" not in prompt.lower()   # blind to its own dials
    assert llm.asked[0][1]["is_background"] is True and llm.asked[0][0]["temperature"] == 0.0
    row = V.rows(home)[-1]
    assert row["observed"] == {"warmth": 0, "humour": 1, "directness": -1}
    assert row["dials"] == {"warmth": -1, "humour": 2, "directness": 0} and row["replies"] == 4
    assert row["date"] == C._mood_day(NOW)
    assert asyncio.run(V.look(home, llm, now=NOW)) == "" and len(llm.asked) == 1            # once a day


@pytest.mark.parametrize("raw,want", [
    ("0 1 -1", (0, 1, -1)), ("+2, -2, 0.", (2, -2, 0)), ("  1/0/−1 ", (1, 0, -1)), ("warmth: 0 0 0", (0, 0, 0)),
    ("1 2 3", None), ("0 1", None), ("0 1 -1 2", None), ("nice\n0 1 -1", None), ("", None), ("a little dry", None),
    ("0.5 1 1", None), (None, None)])
def test_only_three_whole_numbers_are_an_answer(raw, want):
    got = V.parse_answer(raw)
    assert got == (dict(zip(C.DIALS, want)) if want else None)


def test_too_few_replies_no_model_or_a_bad_answer_write_nothing(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _replies(monkeypatch, n=V.MIN_REPLIES - 1)
    llm = _LLM()
    assert asyncio.run(V.look(home, llm, now=NOW)) == "" and llm.asked == []
    _replies(monkeypatch)
    assert asyncio.run(V.look(home, None, now=NOW)) == ""
    assert asyncio.run(V.look(home, _LLM("I think 1 and 2"), now=NOW)).startswith("no look — the answer was not")
    assert V.rows(home) == [] and V.line(home) == "" and V.votes(home) == {}


def test_a_look_votes_only_when_the_replies_are_far_from_the_dial():
    # two steps off, or on the other side of 0: a vote toward what the replies showed
    assert [V.vote(o, d) for o, d in ((0, 2), (2, 0), (-1, 1), (1, -1), (-2, 0))] == [-1, 1, -1, 1, -1]
    # one step off, or the same: none — a mild trait is faint, and voting on it walks every dial to 0
    assert [V.vote(o, d) for o, d in ((0, 1), (1, 0), (1, 2), (0, 0), (-1, -2), (2, 2))] == [0, 0, 0, 0, 0, 0]
    assert V.vote("x", None) == 0 and V.vote(9, -9) == 1


def test_votes_are_the_looks_against_the_dials_as_they_were(tmp_path, monkeypatch):
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    home = _home(tmp_path)
    for date, obs, then in (("2026-11-10", {"warmth": 1, "humour": 0, "directness": 0}, {"warmth": -1, "humour": 1, "directness": 0}),
                            ("2026-11-11", {"warmth": 0, "humour": 0, "directness": 0}, {"warmth": 0, "humour": 2, "directness": 0}),
                            ("2026-11-12", {"warmth": 0, "humour": 2, "directness": 1}, {"warmth": 0, "humour": 2, "directness": 0})):
        V._append(home, {"at": date + "T10:00:00Z", "date": date, "observed": obs, "dials": then, "replies": 5})
    V._append(home, {"at": "x", "date": "not a date", "observed": {"warmth": 2}, "dials": {}})
    V._append(home, {"at": "x", "date": "2026-11-13", "observed": "nope"})
    assert V.votes(home) == {"2026-11-10": {"warmth": 1}, "2026-11-11": {"humour": -1}}
    assert V.line(home) == ("My own look at my replies on 2026-11-12 (5 replies): I meant to be neither diplomatic "
                            "nor blunt and sounded straight about what I think. The rest matched.")
    V._append(home, {"at": "2026-11-14T10:00:00Z", "date": "2026-11-14", "observed": {"warmth": 0, "humour": 2, "directness": 0},
                     "dials": {"warmth": 0, "humour": 2, "directness": 0}, "replies": 6})
    assert V.line(home) == "My own look at my replies on 2026-11-14: they sounded the way my dials say (6 replies)."


def test_what_the_look_saw_is_in_the_view_and_the_introspect_line(tmp_path):
    home = _home(tmp_path)
    assert "My own look" not in C.view(home) and "My own look" not in C.line(home)
    V._append(home, {"at": "2026-11-12T10:00:00Z", "date": "2026-11-12", "observed": {"warmth": 1, "humour": 1, "directness": 0},
                     "dials": dict(C.START), "replies": 4})
    assert "I meant to be a little dry and sounded quietly warm" in C.view(home)
    assert "I meant to be a little dry and sounded quietly warm" in C.line(home)
    assert "how my days go" in C.line(home) and "how your days go" in C.RULE


def test_the_loop_a_run_of_looks_moves_a_dial_toward_what_the_replies_show(tmp_path, monkeypatch):
    """dials → sentence → replies → the look → the dials."""
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    monkeypatch.setattr(C, "day_pulls", lambda days: {})                # the days themselves say nothing
    home = _home(tmp_path)
    C.set_dial(home, "humour", 2, now=datetime.datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    days = [{"date": f"2026-11-{d:02d}"} for d in (1, 2, 3, 4)]
    for d in days[:3]:
        V._append(home, {"at": d["date"] + "T10:00:00Z", "date": d["date"],
                         "observed": {"warmth": -1, "humour": 0, "directness": 0}, "dials": C.load(home)["dials"], "replies": 5})
    said = C.daily(home, days, now=NOW, votes=V.votes(home))
    assert [x.split(":")[0] for x in said] == ["humour +2 → +1"]
    assert "its own look at its replies counted on 3 of those days" in said[0]
    # one step from the dial now: the same looks say nothing more, and the dial stays
    V._append(home, {"at": "2026-11-05T10:00:00Z", "date": "2026-11-05",
                     "observed": {"warmth": -1, "humour": 0, "directness": 0}, "dials": C.load(home)["dials"], "replies": 5})
    assert V.votes(home).get("2026-11-05") is None


@pytest.mark.asyncio
async def test_the_idle_phase_looks_once_and_hands_the_votes_to_the_rule(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _replies(monkeypatch)
    llm = _LLM("0 0 0")
    seen = {}
    real = C.daily
    monkeypatch.setattr(C, "daily", lambda h, days, **k: seen.update(votes=k.get("votes")) or real(h, days, **k))
    agent = A.GhostAgent.__new__(A.GhostAgent)
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=llm, args=None, self_model=None)
    ran, notes = [], {}
    await agent._character_phase(home, ran, notes, ctx)
    assert len(llm.asked) == 1 and ran == ["character"] and notes["character"].startswith("looked at 4 replies")
    assert seen["votes"] == {} and len(V.rows(home)) == 1
    ran2, notes2 = [], {}
    await agent._character_phase(home, ran2, notes2, ctx)
    assert len(llm.asked) == 1 and ran2 == []                           # today's look is there


@pytest.mark.asyncio
async def test_a_deferred_or_failed_look_never_stops_the_days_or_the_rule(tmp_path, monkeypatch):
    from ghost_agent.core.llm import BackgroundDeferred
    home = _home(tmp_path)
    _replies(monkeypatch)
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent._last_mirror_try_at = datetime.datetime.now()
    ran, notes = [], {}
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=_LLM(error=BackgroundDeferred("busy")),
                          args=None, self_model=None)
    await agent._character_phase(home, ran, notes, ctx)
    assert "deferred" in notes["character"] and agent._last_mirror_try_at == datetime.datetime.min
    ctx.llm_client = _LLM(error=RuntimeError("boom"))
    ran, notes = [], {}
    await agent._character_phase(home, ran, notes, ctx)
    assert "the look failed (RuntimeError)" in notes["character"] and V.rows(home) == []


@pytest.mark.asyncio
async def test_with_the_character_off_nobody_looks(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _replies(monkeypatch)
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    llm = _LLM()
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=llm, args=None, self_model=None)
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, [], {}, ctx)
    assert llm.asked == [] and V.rows(home) == []


# ── tastes ──────────────────────────────────────────────────────────────────

from ghost_agent.selfhood import tastes as T  # noqa: E402


def test_a_turn_counts_once_for_each_kind_of_work_it_was():
    got = T.count_turn(["web_search", "browser", "execute", "introspect"], failed_calls=0, calls=4)
    assert got == {"coding": (1.0, 0.0), "research": (1.0, 0.0)}
    assert T.count_turn(["introspect"], 0, 1) == {}                         # not work: no kind
    assert T.count_turn(["web_search"], 1, 2) == {"research": (0.0, 1.0)}  # half the calls failed: bad
    assert T.count_turn(["web_search"], 1, 3) == {"research": (1.0, 0.0)}
    assert T.count_turn(["web_search"], 0, 1, delivered=False) == {"research": (0.0, 1.0)}
    assert T.count_turn(["web_search"], 0, 1, thumb=1) == {"research": (1.0 + T.THUMB_WEIGHT, 0.0)}
    assert T.count_turn(["web_search"], 0, 1, thumb=-1) == {"research": (1.0, T.THUMB_WEIGHT)}
    assert T.count_turn(["web_search"], "x", None) == {"research": (1.0, 0.0)}   # junk counts: no crash


def test_what_was_counted_fades_with_the_days_gone():
    s = T.fold({"kinds": {}, "through": ""}, "2026-11-01", [{"names": ["recall"], "failed": 0, "calls": 1}] * 4)
    assert s == {"kinds": {"memory": {"good": 4.0, "bad": 0.0}}, "through": "2026-11-01"}
    later = T.fold(s, "2026-12-01", [], days_since=int(T.HALF_LIFE_DAYS))
    assert later["kinds"]["memory"]["good"] == pytest.approx(2.0) and later["through"] == "2026-12-01"
    gone = T.fold(s, "2027-11-01", [], days_since=400)
    assert gone["kinds"] == {}                                              # faded below a trace: dropped


def _state(**kinds):
    return {"kinds": {k.replace("_", " "): {"good": g, "bad": b} for k, (g, b) in kinds.items()}, "through": "2026-11-11"}


def test_a_taste_stands_out_only_against_the_agents_own_average():
    s = _state(research=(60, 1), coding=(40, 4), pictures=(10, 6), memory=(25, 10))
    got = T.standing(s)
    assert got["liked"] == ["research"] and got["rough"] == ["pictures", "memory"]
    assert T.standing(_state(research=(60, 0), coding=(40, 0))) == {"liked": [], "rough": []}   # nothing went badly
    assert T.standing(_state(research=(5, 0), pictures=(2, 2))) == {"liked": [], "rough": []}   # too little
    few = T.standing(_state(research=(60, 1), coding=(40, 4), pictures=(3, 4)))
    assert "pictures" not in few["rough"]                                  # under MIN_TURNS: no taste
    # rough needs the floor too: when almost nothing goes badly, twice almost nothing is not rough
    assert T.standing(_state(research=(1000, 1), coding=(100, 0.3)))["rough"] == []


def test_a_reply_hears_only_what_is_liked_and_the_view_says_both(tmp_path):
    home = _home(tmp_path)
    assert T.liked_line(home) == "" and T.line(home) == ""
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(
        _state(research=(60, 1), the_system=(30, 0), pictures=(14, 4), memory=(30, 6))))
    heard = T.liked_line(home)
    assert "research and looking after the system" in heard and "never a reason to choose a tool" in heard
    assert "picture" not in heard and "memory" not in heard and "rough" not in heard
    assert T.line(home) == "Lately: I have come to enjoy research and looking after the system."   # the default
    assert T.line(home, liked_only=False) == ("Lately: I have come to enjoy research and looking after the system; "
                            "making and reading pictures and memory work have been rough.")
    text, stamp = C.section(home)
    assert heard in text and "rough" not in text and stamp == C.stamp(C.load(home)["dials"])
    # review r2: the view is a system REPLY, which goes back to the model as history — liked only
    assert "come to enjoy research" in C.view(home) and "rough" not in C.view(home)


def test_the_section_memo_is_keyed_on_the_tastes_file_too(tmp_path):
    home = _home(tmp_path)
    before, _ = C.section(home)
    assert "enjoy" not in before
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(research=(60, 1), coding=(40, 4),
                                                                                   memory=(30, 6))))
    after, _ = C.section(home)
    assert "Lately you have come to enjoy research" in after


@pytest.mark.parametrize("raw", ["{", "[]", '{"kinds": 3}', '{"kinds": {"research": {"good": "x"}}}',
                                 '{"kinds": {"research": {"good": -5, "bad": 1}}, "through": "nope"}'])
def test_a_bad_tastes_file_is_no_taste_and_never_breaks_the_character(tmp_path, raw):
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(raw)
    assert T.load(home)["kinds"] == {} and T.load(home)["through"] == ""
    assert T.liked_line(home) == "" and C.section(home)[0]


def _turn(at, names, *, failed=0, tid="t", calls=None):
    calls = calls if calls is not None else len(names)
    tc = [{"name": n, "error": "boom" if i < failed else ""} for i, n in enumerate(names)]
    return {"id": tid, "timestamp": at, "tool_calls": tc[:calls] if calls <= len(tc) else tc}


def test_closing_days_reads_owner_turns_and_thumbs_once_and_is_idempotent(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    good = [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a"),
            _turn("2026-11-10T13:00:00Z", ["web_search", "execute"], failed=2, tid="b"),
            _turn("2026-11-11T12:00:00Z", ["recall"], tid="c")]
    bad = [_turn("2026-11-11T13:00:00Z", ["image_generation"], tid="d")]
    reads = []
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: reads.append(window) or (good, bad))
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"trajectory_id": "a", "source": "human_feedback:web", "outcome": "passed"},
        {"trajectory_id": "c", "source": "human_feedback:slack:member", "outcome": "failed"},   # a member's thumb
        {"trajectory_id": "c", "source": "auto", "outcome": "failed"})) + "\nnot json\n")
    done = T.close_days(home, ["2026-11-10", "2026-11-11", "bogus"], tz=UTC)
    assert done == ["2026-11-10", "2026-11-11"] and len(reads) == 1
    k = T.load(home)["kinds"]
    fade = 0.5 ** (1 / T.HALF_LIFE_DAYS)
    assert k["research"]["good"] == pytest.approx((1 + T.THUMB_WEIGHT) * fade, abs=1e-3)
    assert k["research"]["bad"] == pytest.approx(1 * fade, abs=1e-3)       # b: both calls failed
    assert k["coding"]["bad"] == pytest.approx(fade, abs=1e-3)
    assert k["memory"] == {"good": 1.0, "bad": 0.0}                        # only the owner's thumbs count
    assert k["pictures"] == {"good": 0.0, "bad": 1.0}                      # never delivered
    assert T.load(home)["through"] == "2026-11-11"
    assert T.close_days(home, ["2026-11-10", "2026-11-11"], tz=UTC) == [] and len(reads) == 1


def test_closing_days_never_raises_and_writes_nothing_on_a_failure(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)

    def boom(sysd, window):
        raise RuntimeError("no trajectories")
    monkeypatch.setattr(FC, "_owner_turns", boom)
    assert T.close_days(home, ["2026-11-10"], tz=UTC) == []
    assert not (home / "system" / "selfhood" / "tastes.json").exists()
    assert T.close_days(None, ["2026-11-10"]) == []


def test_a_first_run_folds_at_most_the_backfill(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: ([], []))
    start = datetime.date(2026, 6, 1)
    dates = [(start + datetime.timedelta(days=i)).isoformat() for i in range(T.BACKFILL_DAYS + 20)]
    done = T.close_days(home, dates, tz=UTC)
    assert len(done) == T.BACKFILL_DAYS and done[-1] == dates[-1]


@pytest.mark.asyncio
async def test_the_idle_phase_counts_the_closed_days_into_tastes(tmp_path, monkeypatch):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    monkeypatch.setattr(D, "load_days", lambda h: [{"date": "2026-11-10"}, {"date": "2026-11-11"}])
    got = []
    monkeypatch.setattr(T, "close_days", lambda h, dates, **k: got.append(list(dates)) or list(dates))
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=None, args=None, self_model=None)
    ran, notes = [], {}
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, ran, notes, ctx)
    assert got == [["2026-11-10", "2026-11-11"]] and "tastes counted for 2 day(s)" in notes.get("character", "")


# ── the character leans on how hard things land ─────────────────────────────

_REAL_LEAN_DIALS = C.lean_dials


@pytest.fixture
def real_lean(monkeypatch):
    """conftest pins the lean's dials at 0 for every test; these tests read the stored dials."""
    monkeypatch.setattr(C, "lean_dials", _REAL_LEAN_DIALS)


def _ev(kind, dp=-0.2, de=0.1):
    return M.Event(dp, de, "why", kind)


@pytest.mark.parametrize("warmth,factor", [(-2, 0.8), (-1, 0.9), (0, 1.0), (1, 1.1), (2, 1.2)])
def test_warmth_scales_what_comes_from_the_owner_both_ways(warmth, factor):
    dials = {"warmth": warmth, "humour": 2, "directness": -2}
    for kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN, M.KIND_CORRECTION, M.KIND_PRAISE, M.KIND_SENTIMENT):
        for dp in (-0.2, 0.2):
            got = C.lean(_ev(kind, dp), dials)
            assert got.dp == pytest.approx(dp * factor) and got.de == pytest.approx(0.1 * factor)
            assert got.kind == kind and got.cause == "why"


def test_humour_lightens_or_weighs_only_the_agents_own_setbacks():
    playful, serious = {"warmth": 0, "humour": 2, "directness": 0}, {"warmth": 0, "humour": -2, "directness": 0}
    assert C.lean(_ev(M.KIND_TURN, -0.3), playful).dp == pytest.approx(-0.24)
    assert C.lean(_ev(M.KIND_TURN, -0.3), serious).dp == pytest.approx(-0.36)
    good = _ev(M.KIND_TURN, 0.3)
    assert C.lean(good, playful) is good and C.lean(good, serious) is good
    assert C.lean(_ev(M.KIND_THUMB_DOWN), {"warmth": 0, "humour": 2, "directness": 0}).dp == pytest.approx(-0.2)


def test_nothing_else_leans_and_the_bound_holds():
    wild = {"warmth": 9, "humour": -9, "directness": 9}
    for kind in (M.KIND_OWNER, M.KIND_QUIET, M.KIND_QUIET_END, M.KIND_DECAY, M.KIND_MANNER, M.KIND_THUMB_OFF):
        ev = _ev(kind)
        assert C.lean(ev, wild) is ev
    assert C.lean(_ev(M.KIND_PRAISE, 0.5), wild).dp == pytest.approx(0.5 * (1 + 2 * C.LEAN_STEP))
    assert C.lean(_ev(M.KIND_TURN, -0.5), wild).dp == pytest.approx(-0.5 * (1 + 2 * C.LEAN_STEP))
    assert C.lean(None, wild) is None


def _sm(home):
    return SelfModel(root=home / "system" / "selfhood")


def test_the_mood_books_the_leaned_size_on_arrival(tmp_path, real_lean):
    home = _home(tmp_path)
    C.set_dial(home, "warmth", 2, now=NOW)
    plain = _home(tmp_path / "plain")
    C.set_dial(plain, "warmth", 0, now=NOW)
    warm, flat = _sm(home), _sm(plain)
    a = warm.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1")
    b = flat.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1")
    assert a is not None and b is not None
    assert a.moved[0] == pytest.approx(b.moved[0] * 1.2, abs=1e-3)
    assert warm._applied["r1"][0] == pytest.approx(a.base_moved[0])          # the ledger holds what moved
    # the same thumb again changes nothing: it was booked at its leaned size
    assert warm.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1") is None


def test_a_bad_turn_lands_lighter_on_a_playful_character(tmp_path, real_lean):
    home, plain = _home(tmp_path / "p"), _home(tmp_path / "s")
    C.set_dial(home, "humour", 2, now=NOW)
    C.set_dial(plain, "humour", 0, now=NOW)
    sig = M.TurnSignals(tool_calls=6, failures=5)
    a = _sm(home).note_owner_turn(sig, now=NOW)
    b = _sm(plain).note_owner_turn(sig, now=NOW)
    assert a.moved[0] < 0 and a.moved[0] == pytest.approx(b.moved[0] * 0.8, abs=1e-3)


def test_no_lean_when_it_is_switched_off_or_the_character_is_off_or_unreadable(tmp_path, monkeypatch, real_lean):
    plain = _home(tmp_path / "f")
    C.set_dial(plain, "warmth", 0, now=NOW)
    base = _sm(plain).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0]
    seen = []
    for env in ({"GHOST_CHARACTER_LEAN": "0"}, {"GHOST_CHARACTER_LEAN": "1", "GHOST_CHARACTER": "0"}):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        h = _home(tmp_path / f"case{len(seen)}")
        C.set_dial(h, "warmth", 2, now=NOW)
        seen.append(_sm(h).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0])
        monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    monkeypatch.setenv("GHOST_CHARACTER_LEAN", "1")
    h = _home(tmp_path / "broken")
    (h / "system" / "selfhood" / C.CHARACTER_FILENAME).write_text("{not json")
    C._MEMO.clear()
    seen.append(_sm(h).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0])
    assert seen == [pytest.approx(base)] * 3


def test_day_records_read_in_date_order_once_each(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    f = home / "system" / "selfhood" / D.DAYS_FILENAME
    f.write_text("".join(json.dumps(r) + "\n" for r in (
        {"date": "2026-11-09", "n": 1}, {"date": "2026-10-01", "n": 2, "seeded": True},
        {"date": "2026-11-09", "n": 3}, {"date": "2026-10-02", "n": 4, "seeded": True})))
    assert [(r["date"], r["n"]) for r in D.load_days(home)] == [("2026-10-01", 2), ("2026-10-02", 4), ("2026-11-09", 1)]
    assert [r["date"] for r in D.load_days(home, limit=1)] == ["2026-11-09"]


# ── review r1 ───────────────────────────────────────────────────────────────

_TASTY = dict(research=(60, 1), the_system=(30, 0), pictures=(14, 4), memory=(30, 6))


def test_the_introspect_line_carries_liked_tastes_only(tmp_path):
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(**_TASTY)))
    said = C.line(home)
    assert "come to enjoy research" in said and "rough" not in said and "picture" not in said
    assert "rough" not in C.view(home)
    from ghost_agent.tools import introspect as I
    assert "rough" not in (I._self_image(str(home / "system" / "memory")) or "")


def test_an_unreadable_answer_is_tried_at_most_twice_a_day_and_the_next_day_looks_again(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _replies(monkeypatch)
    llm = _LLM("I think warmth is 1")
    for _ in range(4):
        asyncio.run(V.look(home, llm, now=NOW))
    assert len(llm.asked) == V.MAX_TRIES_A_DAY and V.rows(home) == [] and V.votes(home) == {}
    llm.answer = "0 1 -1"
    asyncio.run(V.look(home, llm, now=NOW + datetime.timedelta(days=1)))
    assert len(V.rows(home)) == 1
    asyncio.run(V.look(home, llm, now=NOW + datetime.timedelta(days=2)))
    assert len(V.rows(home)) == 2                                        # a look a day, not one ever


def test_a_later_look_of_the_same_day_with_no_vote_cancels_the_earlier_vote(tmp_path, monkeypatch):
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    home = _home(tmp_path)
    dials = {"warmth": -1, "humour": 1, "directness": 0}
    V._append(home, {"at": "x", "date": "2026-11-05", "observed": {"warmth": 1, "humour": 1, "directness": 0},
                     "dials": dials, "replies": 4})
    assert V.votes(home) == {"2026-11-05": {"warmth": 1}}
    V._append(home, {"at": "y", "date": "2026-11-05", "observed": dict(dials), "dials": dials, "replies": 4})
    assert V.votes(home) == {}


def test_the_handhelds_thumbs_count_for_tastes_and_a_scripts_do_not(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: (
        [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a"), _turn("2026-11-10T12:30:00Z", ["recall"], tid="b")], []))
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(
        json.dumps({"trajectory_id": "a", "source": "human_feedback:clockwork", "outcome": "passed"}) + "\n"
        + json.dumps({"trajectory_id": "b", "source": "human_feedback:api", "outcome": "failed"}) + "\n")
    T.close_days(home, ["2026-11-10"], tz=UTC)
    k = T.load(home)["kinds"]
    # §4UH-e2e review: "api" names no client (a script) — the mood refuses it, so tastes do too
    assert k["research"]["good"] == 1 + T.THUMB_WEIGHT and k["memory"]["bad"] == 0


@pytest.mark.parametrize("humour,raw,trouble", [(2, -0.12, True), (-2, -0.09, False)])
def test_trouble_is_judged_on_the_turns_own_size_and_only_the_booking_leans(tmp_path, monkeypatch, real_lean,
                                                                              humour, raw, trouble):
    home = _home(tmp_path)
    C.set_dial(home, "humour", humour, now=NOW)
    monkeypatch.setattr(M, "turn_events", lambda sig: [M.Event(raw, 0.0, "a call failed in this turn", M.KIND_TURN)])
    sm = _sm(home)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=2, failures=1), now=NOW)
    assert after.moved[0] == pytest.approx(raw * (1 - 0.1 * humour), abs=1e-3)   # leaned once, not twice
    assert (sm._trouble_at is not None) is trouble and (sm._own_trouble_at is not None) is trouble


def test_a_turn_that_moves_only_energy_is_not_leaned():
    ev = M.Event(0.0, -0.3, "a long turn", M.KIND_TURN)
    assert C.lean(ev, {"warmth": 2, "humour": 2, "directness": 2}) is ev


@pytest.mark.asyncio
async def test_the_rule_runs_before_the_look(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _replies(monkeypatch)
    order = []
    real_daily, real_look = C.daily, V.look
    monkeypatch.setattr(C, "daily", lambda h, days, **k: order.append("rule") or real_daily(h, days, **k))

    async def look(*a, **k):
        order.append("look")
        return await real_look(*a, **k)
    monkeypatch.setattr(V, "look", look)
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=_LLM("0 0 0"), args=None,
                          self_model=None)
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, [], {}, ctx)
    assert order == ["rule", "look"]


def _journal(home, rows):
    p = home / "system" / "selfhood" / "mood.events.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_days_thumbs_are_the_owners_last_thumb_per_turn_from_the_label_store(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    _journal(home, [{"at": "2026-11-10T12:01:00Z", "kind": M.KIND_THUMB_UP, "dp": 0.25, "de": 0.0, "cause": "x"}] * 3)
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    rows = [("a", "passed", "human_feedback:web", "2026-11-10T10:00:00Z"),
            ("a", "passed", "human_feedback:web", "2026-11-10T11:00:00Z"),      # sent again: one thumb
            ("b", "passed", "human_feedback:clockwork", "2026-11-10T10:00:00Z"),
            ("b", "failed", "human_feedback:clockwork", "2026-11-10T10:01:00Z"),  # changed: the last counts
            ("c", "failed", "human_feedback:slack:member", "2026-11-10T10:00:00Z"),  # a member's
            ("d", "failed", "user_correction", "2026-11-10T10:00:00Z"),          # not a thumb
            ("e", "passed", "human_feedback:api", "2026-11-09T10:00:00Z"),       # the day before
            ("f", "passed", "human_feedback:cli", "2026-11-10T10:00:00Z"),
            ("f", "failed", "human_feedback:cli", "2026-11-11T10:00:00Z")]       # changed the next day
    (tdir / "corrections.jsonl").write_text("".join(json.dumps(
        {"trajectory_id": t, "outcome": o, "source": src, "timestamp": ts}) + "\n" for t, o, src, ts in rows))
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC, now=NOW)
    assert (rec["thumbs_up"], rec["thumbs_down"]) == (1, 1)                # the journal's three rows are not thumbs
    assert D.owner_thumbs(home, datetime.date(2026, 11, 11), tz=UTC) == (0, 1)
    assert D.owner_thumbs(home, datetime.date(2026, 11, 9), tz=UTC) == (0, 0)       # "api" is not the owner's hand
    assert D.owner_thumbs(None, datetime.date(2026, 11, 11)) == (0, 0)

def _pulls_for(monkeypatch, prev, today):
    vals = list(prev) + [today]
    days = [{"date": (datetime.date(2026, 10, 1) + datetime.timedelta(days=i)).isoformat(), "v": v}
            for i, v in enumerate(vals)]
    monkeypatch.setattr(C, "day_signals", lambda rec: {"warmth": None, "humour": None, "directness": rec["v"]})
    return C.day_pulls(days).get(days[-1]["date"], {}).get("directness")


def test_a_pull_is_measured_in_the_agents_own_spread(monkeypatch):
    prev = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]                    # mean 2, spread 2
    assert _pulls_for(monkeypatch, prev, 2.8) is None        # 0.4 deviations (0.8 unscaled would pull)
    assert _pulls_for(monkeypatch, prev, 3.2) == 1
    assert _pulls_for(monkeypatch, prev, 0.8) == -1


def test_the_band_is_strict_and_needs_enough_days(monkeypatch):
    prev = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]                    # mean 1, spread 1
    assert _pulls_for(monkeypatch, prev, 1.5) is None        # exactly 0.5 deviations: no pull
    assert _pulls_for(monkeypatch, prev, 0.5) is None
    assert _pulls_for(monkeypatch, prev, 1.5001) == 1
    assert _pulls_for(monkeypatch, prev[:C.BASE_MIN - 1], 9.0) is None
    assert _pulls_for(monkeypatch, [0.0, 2.0, 0.0, 2.0, 0.0][:C.BASE_MIN], 9.0) == 1


def test_liked_is_at_most_half_the_average_bad_share_exactly():
    # average bad share 10/200 = 0.05; research 1/40 = 0.025 is exactly half of it: liked
    s = {"kinds": {"research": {"good": 39.0, "bad": 1.0}, "coding": {"good": 151.0, "bad": 9.0}}, "through": "x"}
    assert T.standing(s)["liked"] == ["research"]
    s["kinds"]["research"] = {"good": 38.9, "bad": 1.1}
    s["kinds"]["coding"] = {"good": 151.1, "bad": 8.9}       # same average, research just over half
    assert T.standing(s)["liked"] == []


def test_rough_is_at_least_twice_the_average_exactly():
    # average 20/400 = 0.05, twice it 0.1 (over the 0.08 floor); pictures 10/100 = 0.1 exactly: rough
    s = {"kinds": {"pictures": {"good": 90.0, "bad": 10.0}, "coding": {"good": 290.0, "bad": 10.0}}, "through": "x"}
    assert T.standing(s)["rough"] == ["pictures"]
    s["kinds"]["pictures"] = {"good": 90.1, "bad": 9.9}
    s["kinds"]["coding"] = {"good": 289.9, "bad": 10.1}
    assert T.standing(s)["rough"] == []


def test_humour_needs_three_turns_of_work():
    assert C.day_signals({"own": {"clean_turns": 2, "trouble": 1}})["humour"] == pytest.approx(0.0)
    assert C.day_signals({"own": {"clean_turns": 2, "trouble": 0}})["humour"] is None


def test_no_taste_until_there_are_twice_min_turns_in_all():
    # research alone is enough turns for a taste, but all kinds together are under 2 × MIN_TURNS
    s = {"kinds": {"research": {"good": 8.0, "bad": 0.0}, "coding": {"good": 2.0, "bad": 3.0}}, "through": "x"}
    assert T.standing(s) == {"liked": [], "rough": []}
    s["kinds"]["coding"] = {"good": 5.0, "bad": 3.0}
    assert T.standing(s)["liked"] == ["research"]



def test_a_look_whose_call_fails_is_one_of_the_days_tries(tmp_path, monkeypatch):
    from ghost_agent.core.llm import BackgroundDeferred
    home = _home(tmp_path)
    _replies(monkeypatch)
    llm = _LLM(error=RuntimeError("timeout"))
    for _ in range(4):
        with pytest.raises(RuntimeError):
            asyncio.run(V.look(home, llm, now=NOW))
        if V.looked_today(home, NOW):
            break
    assert len(llm.asked) == V.MAX_TRIES_A_DAY
    deferred = _LLM(error=BackgroundDeferred("busy"))
    h2 = _home(tmp_path / "d")
    for _ in range(3):
        with pytest.raises(BackgroundDeferred):
            asyncio.run(V.look(h2, deferred, now=NOW))
    assert len(deferred.asked) == 3 and not V.looked_today(h2, NOW)     # a deferral is not a try



@pytest.mark.asyncio
async def test_the_rough_tastes_are_said_in_the_log_when_days_are_counted(tmp_path, monkeypatch):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(**_TASTY)))
    monkeypatch.setattr(D, "load_days", lambda h: [{"date": "2026-11-10"}])
    monkeypatch.setattr(T, "close_days", lambda h, dates, **k: ["2026-11-10"])
    logged = []
    monkeypatch.setattr(A, "pretty_log", lambda title, msg="", **k: logged.append((title, msg)))
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=None, args=None, self_model=None)
    notes = {}
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, [], notes, ctx)
    assert any(t == "Tastes" and "rough" in m for t, m in logged)
    assert "rough" not in notes.get("character", "")


def test_only_a_thumbs_outcome_counts(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(json.dumps({"trajectory_id": "a", "outcome": "unknown",
                                                        "source": "human_feedback:web",
                                                        "timestamp": "2026-11-10T10:00:00Z"}) + "\n")
    assert D.owner_thumbs(home, datetime.date(2026, 11, 10), tz=UTC) == (0, 0)


# ── the `mirror` command shows the look (§4UH-m) ─────────────────────────────

def _sv_rows(home, rows):
    p = home / "system" / "selfhood" / V.LOG_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_the_mirror_command_shows_each_look_meant_against_sounded_and_its_pull(tmp_path, monkeypatch):
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    from ghost_agent.selfhood import mirror as MI
    home = tmp_path
    then = {"warmth": -1, "humour": 1, "directness": 0}
    _sv_rows(home, [
        {"at": "2026-11-12T03:00:00Z", "date": "2026-11-12", "failed": True, "replies": 4},     # a later look: hidden
        {"at": "2026-11-12T04:00:00Z", "date": "2026-11-12", "observed": {"warmth": 1, "humour": 1, "directness": 0},
         "dials": then, "replies": 4},
        {"at": "2026-11-13T04:00:00Z", "date": "2026-11-13", "observed": {"warmth": 0, "humour": 0, "directness": 2},
         "dials": then, "replies": 5},
        {"at": "2026-11-14T03:00:00Z", "date": "2026-11-14", "failed": True, "replies": 3, "error": "TimeoutError"},
        {"at": "2026-11-14T04:00:00Z", "date": "2026-11-14", "failed": True, "replies": 3},     # one line a day
        {"date": "nonsense", "observed": {}},
    ])
    got = V.history_lines(home)
    assert got == [
        "- **2026-11-12** · 4 replies — warmth meant -1, sounded +1 · humour meant +1, sounded +1 · "
        "directness meant 0, sounded 0 → votes warmth up",
        "- **2026-11-13** · 5 replies — warmth meant -1, sounded 0 · humour meant +1, sounded 0 · "
        "directness meant 0, sounded +2 → votes directness up",
        "- **2026-11-14** · 3 replies — no look: the call failed"]
    assert V.history_lines(home, limit=1) == got[-1:] and V.history_lines(home, limit=0) == []
    b = MI.mirror_banner(home)
    assert "My character now: «" in b and "\n".join(got) in b and "The latest looks:" in b
    assert V.line(home) in b                                    # the latest look in words, under the list
    assert "the Mirror was a diary" not in b                    # no diary: no history paragraph
    assert b.index("My character now") < b.index("The latest looks")


def test_the_mirror_command_keeps_the_old_diary_below_as_history(tmp_path):
    from ghost_agent.selfhood import mirror as MI
    home = tmp_path
    p = home / MI.MIRROR_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": "2026-11-09", "sentence": "I am bright.", "prev": MI.SEED_SENTENCE,
                             "replies": [], "rest": [1, -1]}) + "\n", encoding="utf-8")
    b = MI.mirror_banner(home)
    assert "No look yet" in b and "**Until 2026-11-09**" in b and "— I am bright." in b
    assert b.index("No look yet") < b.index("Until 2026-11-09")
    # §4UH-e2e: the diary's rest is not shown — the character's dials set it, and at the starting dials it is even
    assert "Where my mood settles" not in b
    assert "The latest look (" not in b and "skipped" not in b   # no quoted replies, no skip reason


def test_the_mirror_command_with_an_unreadable_character_says_so(tmp_path):
    from ghost_agent.selfhood import mirror as MI
    p = tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert "cannot be read" in MI.mirror_banner(tmp_path)


def test_a_broken_look_history_falls_back_to_the_diary(tmp_path, monkeypatch):
    from ghost_agent.selfhood import mirror as MI
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(V, "history_lines", boom)
    b = MI.mirror_banner(tmp_path)
    assert b.startswith("**The Mirror**") and "could not be read just now" in b and "diary" not in b
    # review: the fallback is NOT the diary's banner (it quotes replies and says replies hear its sentence)
    p = tmp_path / MI.MIRROR_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": "2026-11-09", "sentence": "I am bright.", "prev": MI.SEED_SENTENCE,
                             "replies": [{"at": "2026-11-09T10:00:00", "excerpt": "SECRET REPLY"}]}) + "\n")
    b = MI.mirror_banner(tmp_path)
    assert "— I am bright." in b and "SECRET REPLY" not in b and "before every reply" not in b


def test_the_mirror_command_shows_the_latest_seven_of_many(tmp_path):
    from ghost_agent.selfhood import mirror as MI
    then = {"warmth": 0, "humour": 0, "directness": 0}
    _sv_rows(tmp_path, [{"at": f"2026-10-{d:02d}T04:00:00Z", "date": f"2026-10-{d:02d}", "replies": True if d == 9 else 3,
                         "observed": then, "dials": then} for d in range(1, 10)])
    got = V.history_lines(tmp_path)
    assert len(got) == V.SHOW_LAST == 7 and got[0].startswith("- **2026-10-03**") and got[-1].startswith("- **2026-10-09**")
    assert "replies" not in got[-1] and "· 3 replies" in got[0]          # a bool count is not a count
    assert "(some replies)" in V.line(tmp_path)                           # nor in the line under the list
    p = tmp_path / MI.MIRROR_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"date": f"2026-10-{d:02d}", "sentence": f"I am number {d}.", "replies": []}) + "\n"
                         for d in range(1, 10)))
    b = MI.mirror_banner(tmp_path)
    assert "I am number 9." in b and "I am number 3." in b and "I am number 2." not in b and "the last 7 shown" in b


def test_the_mirror_command_says_how_many_replies_a_first_look_needs(tmp_path, monkeypatch):
    from ghost_agent.selfhood import mirror as MI
    monkeypatch.setattr(V, "MIN_REPLIES", 5)
    assert "at least 5 replies" in MI.mirror_banner(tmp_path)


def test_todays_failed_try_is_not_a_missed_day_while_a_second_may_come(tmp_path):
    now = datetime.datetime(2026, 11, 14, 12, 0, tzinfo=datetime.timezone.utc)
    day = C._mood_day(now)
    _sv_rows(tmp_path, [{"at": "x", "date": day, "failed": True, "replies": 3}])
    assert V.history_lines(tmp_path, now=now) == []
    _sv_rows(tmp_path, [{"at": "x", "date": day, "failed": True, "replies": 3}] * V.MAX_TRIES_A_DAY)
    assert V.history_lines(tmp_path, now=now) == [f"- **{day}** · 3 replies — no look: the answer could not be read"]


def test_a_vote_on_a_pinned_dial_says_it_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    C.set_pin(tmp_path, "humour", True)
    then = {"warmth": 0, "humour": 0, "directness": 0}
    _sv_rows(tmp_path, [{"at": "2026-11-12T04:00:00Z", "date": "2026-11-12", "replies": 3,
                         "observed": {"warmth": 2, "humour": -2, "directness": 0}, "dials": then}])
    assert V.history_lines(tmp_path)[0].endswith("→ votes warmth up, humour down (pinned: not counted)")


def test_the_picker_reads_a_looks_utc_stamp(tmp_path):
    from ghost_agent.selfhood import mirror as MI
    # review: py3.10 refused the "Z" → 0.0 → the plain 48 h window, re-reading replies the last look rated
    assert MI._look_epoch("2026-11-11T21:27:50Z") == datetime.datetime(2026, 11, 11, 21, 27, 50,
                                                                        tzinfo=datetime.timezone.utc).timestamp()
    assert MI._look_epoch("nonsense") == 0.0 and MI._look_epoch(None) == 0.0


def test_a_vote_from_before_the_dials_last_change_says_it_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv(V.VOTE_ENV, "1")              # §4UH-e2e: votes are behind a switch
    then = {"warmth": 0, "humour": 0, "directness": 0}
    _sv_rows(tmp_path, [{"at": f"2026-11-{d}T04:00:00Z", "date": f"2026-11-{d}", "replies": 3,
                         "observed": {"warmth": 0, "humour": 2, "directness": 0}, "dials": then} for d in (10, 13)])
    C.set_pin(tmp_path, "humour", True, now=datetime.datetime(2026, 11, 11, 12, tzinfo=datetime.timezone.utc))
    C.set_pin(tmp_path, "humour", False, now=datetime.datetime(2026, 11, 12, 12, tzinfo=datetime.timezone.utc))
    got = V.history_lines(tmp_path)
    assert got[0].endswith("→ votes humour up (the dial changed since: not counted)")   # used up by the unpin
    assert got[1].endswith("→ votes humour up")                                          # after it: counts


def test_a_first_failed_try_today_says_a_second_one_comes(tmp_path):
    from ghost_agent.selfhood import mirror as MI
    _sv_rows(tmp_path, [{"at": "x", "date": C._mood_day(), "failed": True, "replies": 5}])
    b = MI.mirror_banner(tmp_path)
    assert "one more try comes" in b and "at least 3 replies" not in b
    _sv_rows(tmp_path, [{"at": "x", "date": C._mood_day(), "failed": True, "replies": 5}] * 2)
    assert "one more try comes" not in MI.mirror_banner(tmp_path)


# ── §4UH-e2e: the look is shown, not a vote; the dials set where the mood rests ─────────────────────

def test_with_the_vote_off_a_look_leans_but_moves_nothing(tmp_path, monkeypatch):
    from ghost_agent.selfhood import mirror as MI
    monkeypatch.delenv(V.VOTE_ENV, raising=False)
    assert V.vote_enabled() is False
    then = {"warmth": -1, "humour": 1, "directness": 0}
    _sv_rows(tmp_path, [{"at": "2026-11-11T04:00:00Z", "date": "2026-11-11", "replies": 6,
                         "observed": {"warmth": 1, "humour": 1, "directness": 1}, "dials": then}])
    assert V.votes(tmp_path) == {}                                   # nothing reaches `character.propose`
    assert V.history_lines(tmp_path)[0].endswith("→ leans warmth up (not a vote)")
    b = MI.mirror_banner(tmp_path)
    assert "not what moves me" in b and "adds one vote" not in b and "and by your commands" in b
    monkeypatch.setenv(V.VOTE_ENV, "1")
    assert V.votes(tmp_path) == {"2026-11-11": {"warmth": 1}} and "adds one vote" in MI.mirror_banner(tmp_path)


@pytest.mark.parametrize("dials, units", [
    ({"warmth": -1, "humour": 1, "directness": 0}, (0, 0)),          # the start: even
    ({"warmth": 0, "humour": 1, "directness": 1}, (1, 1)),           # live 11-12: a little sunny and brisk
    ({"warmth": -1, "humour": 1, "directness": 2}, (0, 0)),          # blunt is not a mood: no "restless" rest
    ({"warmth": 2, "humour": -1, "directness": 0}, (1, -1)),         # warm + serious: a little sunny and calm
    ({"warmth": 2, "humour": 2, "directness": 2}, (2, 2)),
    ({"warmth": -2, "humour": -2, "directness": -2}, (0, 0)),        # dry + serious is not gloomy
    ({"warmth": 1, "humour": 0, "directness": 0}, (1, 0)),
    ({"warmth": 9, "humour": "x"}, (1, 0)),                          # brought into range
])
def test_the_dials_set_where_the_mood_rests(dials, units):
    assert C.rest_units(dials) == units


@pytest.mark.parametrize("w", range(-2, 3))
@pytest.mark.parametrize("h", range(-2, 3))
@pytest.mark.parametrize("d", range(-2, 3))
def test_a_rest_never_sits_in_an_unhappy_restless_or_bored_region(w, h, d):
    from ghost_agent.selfhood import mood as MM
    from ghost_agent.selfhood import mirror as MI
    a, b = C.rest_units({"warmth": w, "humour": h, "directness": d})
    assert MM.rest_label((a * MI.REST_UNIT, b * MI.REST_UNIT)) in ("neutral", "content", "cheerful", "calm", "curious")


def test_the_rest_pair_follows_the_character_and_its_switches(tmp_path, monkeypatch):
    from ghost_agent.selfhood import mirror as MI
    C.set_dial(tmp_path, "directness", 1)
    assert C.rest_pair(tmp_path) == (0.0, 0.0)                       # the start + blunt: even
    C.set_dial(tmp_path, "warmth", 2)
    C.set_dial(tmp_path, "humour", 2)
    assert C.rest_pair(tmp_path) == (2 * MI.REST_UNIT, 2 * MI.REST_UNIT)
    assert "Where my mood settles when nothing happens (set by my character)" in MI.mirror_banner(tmp_path)
    monkeypatch.setenv(MI.REST_ENV, "0")
    assert C.rest_pair(tmp_path) == (0.0, 0.0) and "Where my mood settles" not in MI.mirror_banner(tmp_path)
    monkeypatch.delenv(MI.REST_ENV)
    monkeypatch.setenv("GHOST_CHARACTER", "0")                       # off: the Mirror's (no diary: even)
    assert C.rest_pair(tmp_path) == MI.rest_pair(tmp_path) == (0.0, 0.0)
    monkeypatch.delenv("GHOST_CHARACTER")
    p = tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.write_text("{broken", encoding="utf-8")
    assert C.rest_pair(tmp_path) == (0.0, 0.0)
    assert C.rest_pair(None) == (0.0, 0.0)


def test_main_attaches_the_characters_rest_to_the_mood():
    """Parsed, not a text pin: the one `attach_rest(...)` call in main.py passes a lambda that calls
    `<character module>.rest_pair(...)`, where that name is imported from `.selfhood.character`."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/ghost_agent/main.py").read_text())
    char_names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                  and n.module == "selfhood" and n.level == 1 for a in n.names if a.name == "character"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "attach_rest"]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Lambda)
    body = calls[0].args[0].body
    assert isinstance(body, ast.Call) and isinstance(body.func, ast.Attribute) and body.func.attr == "rest_pair"
    assert isinstance(body.func.value, ast.Name) and body.func.value.id in char_names


def test_rest_units_round_toward_zero_both_ways_and_a_broken_file_rests_even(tmp_path, monkeypatch):
    assert C.rest_units({"warmth": 1, "humour": 0, "directness": -1}) == (1, 0)       # half rounds away from 0
    assert C.rest_units({"warmth": -2, "humour": -1, "directness": 0}) == (0, 0)      # never below even
    C.set_dial(tmp_path, "warmth", 2)
    C.set_dial(tmp_path, "humour", 2)
    assert C.rest_pair(tmp_path) != (0.0, 0.0)
    # warm starting dials, so a broken file falling back to them would NOT rest even (review r2: else equivalent)
    monkeypatch.setattr(C, "START", {"warmth": 2, "humour": 2, "directness": 2})
    (tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME).write_text("{broken", encoding="utf-8")
    assert C.unreadable(tmp_path) and C.rest_units(C.dials(tmp_path)) != (0, 0)
    assert C.rest_pair(tmp_path) == (0.0, 0.0)                     # not the starting dials' rest


def test_a_look_one_step_off_says_no_lean(tmp_path, monkeypatch):
    monkeypatch.delenv(V.VOTE_ENV, raising=False)
    then = {"warmth": -1, "humour": 1, "directness": 0}
    _sv_rows(tmp_path, [{"at": "2026-11-11T04:00:00Z", "date": "2026-11-11", "replies": 6,
                         "observed": {"warmth": 0, "humour": 1, "directness": 0}, "dials": then}])
    assert V.history_lines(tmp_path)[0].endswith("→ no lean")


def test_a_capitalised_owner_source_counts_as_the_mood_counts_it(tmp_path):
    from ghost_agent.selfhood import days as D
    tdir = tmp_path / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(json.dumps({"trajectory_id": "a", "outcome": "passed",
                                                        "source": " Human_Feedback:Web ",
                                                        "timestamp": "2026-11-10T10:00:00Z"}) + "\n")
    assert D.owner_thumbs(tmp_path, datetime.date(2026, 11, 10), tz=UTC) == (1, 0)
    # … and tastes read the same row as the owner's thumb
    import ghost_agent.core.failure_census as FC
    from unittest import mock
    with mock.patch.object(FC, "_owner_turns", lambda sysd, window: (
            [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a")], [])):
        T.close_days(tmp_path, ["2026-11-10"], tz=UTC)
    assert T.load(tmp_path)["kinds"]["research"]["good"] == 1 + T.THUMB_WEIGHT
